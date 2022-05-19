
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd/colored_kinfu.hpp>

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "example.hpp"         // Include short list of convenience functions for rendering

#include <thread>
#include <queue>
#include <atomic>
#include <fstream>
#include <algorithm>
#include <iostream>

#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::colored_kinfu;

constexpr float MAX_DIST = 2.5f;
constexpr float MIN_DIST = 0.f;

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;
constexpr int WIN_WIDTH = 1280;
constexpr int WIN_HEIGHT = 720;

// Handles all the OpenGL calls needed to display the point cloud
void draw_kinfu_pointcloud(glfw_state& app_state, Mat points, Mat normals, Mat colors)
{
    // Define new matrix which will later hold the coloring of the pointcloud

    // OpenGL commands that prep screen for the pointcloud
    glLoadIdentity();
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
    glClear(GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    gluPerspective(65, 1.3, 0.01f, 10.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

    glTranslatef(0, 0, 1 + app_state.offset_y*0.05f);
    glRotated(app_state.pitch-20, 1, 0, 0);
    glRotated(app_state.yaw+5, 0, 1, 0);
    glTranslatef(0, 0, -0.5f);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glBegin(GL_POINTS);
    // this segment actually prints the pointcloud

    for (int i = 0; i < points.rows; i++)
    {
        // Get point coordinates from 'points' matrix
        float x = points.at<float>(i, 0);
        float y = points.at<float>(i, 1);
        float z = points.at<float>(i, 2);

        // Get point coordinates from 'normals' matrix
        float nx = normals.at<float>(i, 0);
        float ny = normals.at<float>(i, 1);
        float nz = normals.at<float>(i, 2);

        // Get colors from the 'colors' matrix
        uint8_t r = static_cast<uint8_t>(colors.at<float>(i, 0));
        uint8_t g = static_cast<uint8_t>(colors.at<float>(i, 1));
        uint8_t b = static_cast<uint8_t>(colors.at<float>(i, 2));

        // Register color and coordinates of the current point
        glColor3ub(r, g, b);
        glNormal3f(nx, ny, nz);
        glVertex3f(x, y, z);
    }
    // OpenGL cleanup
    glEnd();
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();
}


void export_to_ply(Mat points, Mat normals, Mat colors)
{
    // First generate a filename
    const size_t buffer_size = 50;
    char fname[buffer_size];
    time_t t = time(0);   // get time now
    struct tm * now = localtime(&t);
    strftime(fname, buffer_size, "%m%d%y %H%M%S.ply", now);
    std::cout << "exporting to" << fname << std::endl;

    // Write the ply file
    std::ofstream out(fname);
    out << "ply\n";
    out << "format binary_little_endian 1.0\n";
    out << "comment pointcloud saved from Realsense Viewer\n";
    out << "element vertex " << points.rows << "\n";
    out << "property float" << sizeof(float) * 8 << " x\n";
    out << "property float" << sizeof(float) * 8 << " y\n";
    out << "property float" << sizeof(float) * 8 << " z\n";

    out << "property float" << sizeof(float) * 8 << " nx\n";
    out << "property float" << sizeof(float) * 8 << " ny\n";
    out << "property float" << sizeof(float) * 8 << " nz\n";

    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";
    out.close();

    out.open(fname, std::ios_base::app | std::ios_base::binary);
    for (int i = 0; i < points.rows; i++)
    {
        // write vertices
        out.write(reinterpret_cast<const char*>(&(points.at<float>(i, 0))), sizeof(float));
        out.write(reinterpret_cast<const char*>(&(points.at<float>(i, 1))), sizeof(float));
        out.write(reinterpret_cast<const char*>(&(points.at<float>(i, 2))), sizeof(float));

        // write normals
        out.write(reinterpret_cast<const char*>(&(normals.at<float>(i, 0))), sizeof(float));
        out.write(reinterpret_cast<const char*>(&(normals.at<float>(i, 1))), sizeof(float));
        out.write(reinterpret_cast<const char*>(&(normals.at<float>(i, 2))), sizeof(float));
        
        // write colors
        uint8_t r = static_cast<uint8_t>(colors.at<float>(i, 0));
        uint8_t g = static_cast<uint8_t>(colors.at<float>(i, 1));
        uint8_t b = static_cast<uint8_t>(colors.at<float>(i, 2));
        out.write(reinterpret_cast<const char*>(&r), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&g), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&b), sizeof(uint8_t));
    }
}


// Thread-safe queue for OpenCV's Mat objects
class mat_queue
{
public:
    void push(Mat& item)
    {
        std::lock_guard<std::mutex> lock(_mtx);
        queue.push(item);
    }
    int try_get_next_item(Mat& item)
    {
        std::lock_guard<std::mutex> lock(_mtx);
        if (queue.empty())
            return false;
        item = std::move(queue.front());
        queue.pop();
        return true;
    }
private:
    std::queue<Mat> queue;
    std::mutex _mtx;
};


int main(int argc, char **argv)
{
    // Declare KinFu and params pointers
    Ptr<ColoredKinFu> kf;
    Ptr<Params> params = Params::coloredTSDFParams(false);

    // Create a pipeline and configure it
    rs2::pipeline p;
    rs2::config cfg;
    float depth_scale;
    cfg.enable_stream(RS2_STREAM_DEPTH, WIDTH, HEIGHT, RS2_FORMAT_Z16);
    cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_RGB8);

    auto profile = p.start(cfg);
    auto dev = profile.get_device();
    auto stream_depth = profile.get_stream(RS2_STREAM_DEPTH);
    auto stream_color = profile.get_stream(RS2_STREAM_COLOR);

    // Get a new frame from the camera
    rs2::frameset data = p.wait_for_frames();
    auto d = data.get_depth_frame();
    auto rgb = data.get_color_frame();
    
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            // Set some presets for better results
            dpt.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_DENSITY);
            // Depth scale is needed for the kinfu set-up
            depth_scale = dpt.get_depth_scale();

            // enable the laser
            if (dpt.supports(RS2_OPTION_EMITTER_ENABLED))
                dpt.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f);
            
            // set the laser to max power
            if (dpt.supports(RS2_OPTION_LASER_POWER))
            {
                auto range = dpt.get_option_range(RS2_OPTION_LASER_POWER);
                dpt.set_option(RS2_OPTION_LASER_POWER, range.max);
            }
            break;
        }
    }

    // Declare post-processing filters for better results
    auto decimation = rs2::decimation_filter();
    auto spatial = rs2::spatial_filter();
    auto temporal = rs2::temporal_filter();

    float clipping_dist = MAX_DIST / depth_scale; // convert clipping_dist to raw depth units

    // Use decimation once to get the final size of the frame
    d = decimation.process(d);
    int depth_w = d.get_width();
    int depth_h = d.get_height();
    Size depth_size = Size(depth_w, depth_h);

    int rgb_w = rgb.get_width();
    int rgb_h = rgb.get_height();
    Size rgb_size = Size(rgb_w, rgb_h);

    auto depth_intrin = stream_depth.as<rs2::video_stream_profile>().get_intrinsics();
    auto rgb_intrin = stream_color.as<rs2::video_stream_profile>().get_intrinsics();

    // Configure kinfu's parameters
    params->frameSize = depth_size;
    params->rgb_frameSize = rgb_size;
    
    params->intr = Matx33f(
        depth_intrin.fx,               0, depth_intrin.ppx,
                      0, depth_intrin.fy, depth_intrin.ppy,
                      0,               0,                1);

    params->rgb_intr = Matx33f(
        rgb_intrin.fx,             0, rgb_intrin.ppx,
                    0, rgb_intrin.fy, rgb_intrin.ppy,
                    0,             0,              1);


    params->depthFactor = 1 / depth_scale;

    // Initialize KinFu object
    kf = ColoredKinFu::create(params);

    bool after_reset = false;
    mat_queue points_queue, normals_queue, colors_queue;

    window app(WIN_WIDTH, WIN_HEIGHT, "RealSense KinectFusion Example");
    glfw_state app_state;
    register_glfw_callbacks(app, app_state);

    std::atomic_bool stopped(false);

    // This thread runs KinFu algorithm and calculates the pointcloud by fusing depth data from subsequent depth frames
    std::thread calc_cloud_thread([&]() {
        Mat H_points, H_normals, H_colors;
        try {
            while (!stopped)
            {
                rs2::frameset data = p.wait_for_frames(); // Wait for next set of frames from the camera

                auto d = data.get_depth_frame();
                // Use post processing to improve results
                d = decimation.process(d);
                d = spatial.process(d);
                d = temporal.process(d);
                
                auto rgb = data.get_color_frame();

                // Set depth values higher than clipping_dist to 0, to avoid unnecessary noise in the pointcloud
                uint16_t* p_depth_frame = reinterpret_cast<uint16_t*>(const_cast<void*>(d.get_data()));

                std::replace_if(p_depth_frame, p_depth_frame + depth_w*depth_h, 
                [clipping_dist] (uint16_t d) { return d > clipping_dist; }, 0u);

                // Define matrices on the GPU for KinFu's use
                UMat D_points, D_normals, D_colors;
                
                // Copy frame from CPU to GPU
                Mat H_depth(depth_h, depth_w, CV_16UC1, (void*)d.get_data());
                UMat D_depth(depth_h, depth_w, CV_16UC1);
                H_depth.copyTo(D_depth);
                H_depth.release();

                Mat H_rgb(rgb_h, rgb_w, CV_8UC3, (void*)rgb.get_data());
                UMat D_rgb(rgb_h, rgb_w, CV_8UC3);
                H_rgb.copyTo(D_rgb);
                H_rgb.release();

                bool success = kf->update(D_depth, D_rgb);

                // Run KinFu on the new frame(on GPU)
                if (!success)
                {
                    std::cerr << "Failed to update" << std::endl;
                    kf->reset(); // If the algorithm failed, reset current state
                    // Save the pointcloud obtained before failure
                    export_to_ply(H_points, H_normals, H_colors);

                    // To avoid calculating pointcloud before new frames were processed, set 'after_reset' to 'true'
                    after_reset = true;
                    D_points.release();
                    D_normals.release();
                    D_colors.release();
                }

                // Get current pointcloud
                if (!after_reset)
                {
                    kf->getCloud(D_points, D_normals, D_colors);
                }

                if (!D_points.empty() && !D_normals.empty() && !D_colors.empty())
                {
                    // copy points from GPU to CPU for rendering
                    D_points.copyTo(H_points);
                    D_points.release();
                    D_normals.copyTo(H_normals);
                    D_normals.release();
                    D_colors.copyTo(H_colors);
                    D_colors.release();
                    // Push to queue for rendering
                    points_queue.push(H_points);
                    normals_queue.push(H_normals);
                    colors_queue.push(H_colors);
                }
                after_reset = false;
            }
        }
        catch (const std::exception& e) // Save pointcloud in case an error occurs (for example, camera disconnects)
        {
            std::cerr << e.what() << std::endl;
            export_to_ply(H_points, H_normals, H_colors);
            exit(EXIT_FAILURE);
        }
    });
    std::cerr << "thread creation" << std::endl;
    // Main thread handles rendering of the pointcloud
    Mat points, normals, colors;
    while (app)
    {
        // Get the current state of the pointcloud
        points_queue.try_get_next_item(points);
        normals_queue.try_get_next_item(normals);
        colors_queue.try_get_next_item(colors);
        if (!points.empty() && !normals.empty() && !colors.empty()) // points or normals might not be ready on first iterations
            draw_kinfu_pointcloud(app_state, points, normals, colors);
    }
    stopped = true;
    calc_cloud_thread.join();

    // Save the pointcloud upon closing the app
    export_to_ply(points, normals, colors);

    return 0;
}

