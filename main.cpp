#include <iostream>
#include <string>
#include "yolov8_lib.h"
#include "BYTETracker.h"
#include <thread>
#include <queue>
#include <map>
#include <deque>
#include <set>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <fstream>
#include <locale>

// 定义帧数据结构
struct FrameData {
    cv::Mat frame;
    double timestamp;
};

// 全局变量
std::queue<FrameData> input_queue;  // 输入帧队列
std::queue<FrameData> output_queue; // 输出帧队列
std::mutex input_mutex, output_mutex;
std::condition_variable input_cv, output_cv;
bool is_running = true;

// 存储每个跟踪目标的轨迹历史
std::map<int, std::deque<cv::Point>> trajectories;
// 存储已被计数的跟踪ID
std::set<int> counted_tracks;
// 计数器
int counter = 0;

std::vector<int> trackClasses{0};  // 钢卷

bool isTrackingClass(int class_id) {
    for (auto& c : trackClasses) {
        if (class_id == c) return true;
    }
    return false;
}

// 在图像上绘制垂直虚线和计数信息
void drawCountingLine(cv::Mat& img, int count) {
    // 绘制垂直虚线（在图像中间）
    int line_x = img.cols / 2;
    cv::Scalar lineColor(0, 255, 0); // 绿色虚线
    
    // 创建虚线效果
    for(int y = 0; y < img.rows; y += 5) {
        cv::line(img, cv::Point(line_x, y), cv::Point(line_x, y+3), lineColor, 2);
    }
    
    // 显示计数信息
    cv::putText(img, "Total: " + std::to_string(count), 
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 
                0.6, lineColor, 2, cv::LINE_AA);
}

// 绘制轨迹
void drawTrajectories(cv::Mat& img, const std::map<int, std::deque<cv::Point>>& trajectories, 
                     BYTETracker& tracker) {
    for (const auto& track : trajectories) {
        const auto& points = track.second;
        if (points.size() < 2) continue;
        
        for (size_t i = 1; i < points.size(); i++) {
            // 使用跟踪ID的颜色绘制轨迹线
            cv::line(img, points[i-1], points[i], 
                    tracker.get_color(track.first), 2);
        }
    }
}

// 更新轨迹并检查计数
void updateTrajectories(const std::vector<STrack>& tracks, int line_x) {
    for (const auto& track : tracks) {
        int track_id = track.track_id;
        
        // 获取当前对象的中心点
        std::vector<float> tlwh = track.tlwh;
        int x_center = tlwh[0] + tlwh[2]/2;
        int y_center = tlwh[1] + tlwh[3]/2;
        cv::Point center_point(x_center, y_center);
        
        // 更新轨迹
        if (trajectories.find(track_id) == trajectories.end()) {
            trajectories[track_id] = std::deque<cv::Point>();
        }
        trajectories[track_id].push_back(center_point);
        
        // 限制轨迹长度以防止过度内存使用
        if (trajectories[track_id].size() > 30) {
            trajectories[track_id].pop_front();
        }
        
        // 检查是否穿过计数线
        if (trajectories[track_id].size() >= 2) {
            auto& points = trajectories[track_id];
            cv::Point prev = points[points.size() - 2];
            cv::Point curr = points[points.size() - 1];
            
            // 检查是否穿过垂直线（从右到左或从左到右）
            if ((prev.x > line_x && curr.x <= line_x) || 
                (prev.x <= line_x && curr.x > line_x)) {
                // 如果此ID尚未被计数
                if (counted_tracks.find(track_id) == counted_tracks.end()) {
                    counter++; // 增加计数
                    counted_tracks.insert(track_id); // 标记为已计数
                }
            }
        }
    }
}

// GStreamer总线回调函数
static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = static_cast<GMainLoop *>(data);

    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("end of stream\n");
            g_main_loop_quit(loop);
            break;

        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_free(debug);
            g_printerr("error: %s\n", error->message);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }

    return TRUE;
}

// 读取帧的线程函数
void read_frames(cv::VideoCapture &cap) {
    while (is_running) {
        cv::Mat frame;
        if (cap.read(frame)) {
            double timestamp = cap.get(cv::CAP_PROP_POS_MSEC) * 1000000;
            std::lock_guard<std::mutex> lock(input_mutex);
            if (input_queue.size() < 30) { // 设置队列限制
                input_queue.push({frame, timestamp});
                input_cv.notify_one();
            }
        } else {
            is_running = false;
            input_cv.notify_all();
            output_cv.notify_all();
            break;
        }
    }
}

// 处理帧的线程函数
void process_frames(YoloDetecter &detecter, BYTETracker &tracker, int line_x, bool show_gui) {
    int frame_count = 0;
    int total_ms = 0;
    
    // FPS计算变量
    int fps_frame_count = 0;
    auto fps_start_time = std::chrono::steady_clock::now();
    double current_fps = 0.0;

    while (is_running) {
        // 定义帧数据结构变量
        FrameData frame_data;
        {
            // 创建互斥锁保护共享资源
            std::unique_lock<std::mutex> lock(input_mutex);
            // 等待队列非空或程序停止运行
            input_cv.wait(lock, [] { return !input_queue.empty() || !is_running; });
            // 如果程序停止且队列为空则退出循环
            if (!is_running && input_queue.empty()) break;
            // 获取队列中的第一帧数据
            frame_data = input_queue.front();
            // 从队列中移除已处理的帧
            input_queue.pop();
        }

        frame_count++;
        fps_frame_count++;
        
        // 获取当前帧的副本用于显示和处理
        cv::Mat display_img = frame_data.frame.clone();
        
        // 执行YOLO推理
        auto start = std::chrono::system_clock::now();
        std::vector<DetectResult> res = detecter.inference(display_img);

        // 过滤要跟踪的类别
        std::vector<Object> objects;
        for (long unsigned int j = 0; j < res.size(); j++) {
            cv::Rect r = res[j].tlwh;
            float conf = (float)res[j].conf;
            int class_id = (int)res[j].class_id;

            if (isTrackingClass(class_id)) {
                cv::Rect_<float> rect((float)r.x, (float)r.y, (float)r.width, (float)r.height);
                Object obj{rect, class_id, conf};
                objects.push_back(obj);
            }
        }

        // 执行目标跟踪
        std::vector<STrack> output_stracks = tracker.update(objects);
        
        // 更新轨迹并检查计数
        updateTrajectories(output_stracks, line_x);

        auto end = std::chrono::system_clock::now();
        total_ms = total_ms + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // 绘制跟踪结果
        for (int i = 0; i < output_stracks.size(); i++) {
            std::vector<float> tlwh = output_stracks[i].tlwh;
            if (tlwh[2] * tlwh[3] > 20) {
                cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
                cv::putText(display_img, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5), 
                            0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::rectangle(display_img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
            }
        }
        
        // 绘制轨迹
        drawTrajectories(display_img, trajectories, tracker);
        
        // 绘制计数线和计数信息
        drawCountingLine(display_img, counter);
        
        // 显示帧率和目标数量
        cv::putText(display_img, cv::format("frame: %d  fps: %.2f  count: %ld", 
                    frame_count, current_fps, output_stracks.size()), 
                    cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        // FPS计算
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - fps_start_time).count();
        
        if (elapsed >= 1) {  // 每秒更新一次FPS
            current_fps = static_cast<double>(fps_frame_count) / elapsed;
            std::cout << "fps: " << std::fixed << std::setprecision(2) << current_fps << std::endl;
            
            // 重置计数器
            fps_frame_count = 0;
            fps_start_time = current_time;
        }
        
        // 如果启用GUI，显示处理后的图像
        if (show_gui) {
            cv::namedWindow("tracking result", cv::WINDOW_NORMAL);
            cv::imshow("tracking result", display_img);
            char key = cv::waitKey(1);
            if (key == 27) { // ESC键退出
                is_running = false;
                break;
            }
        }

        {
            std::lock_guard<std::mutex> lock(output_mutex);
            output_queue.push({display_img, frame_data.timestamp});
            output_cv.notify_one();
        }
    }
}

int run(const std::string& input_rtmp, const std::string& output_rtmp, const std::string& modelPath, 
        int trackBuffer, bool show_gui,
        float track_thresh, float high_thresh, float match_thresh)
{
    // 初始化GStreamer
    gst_init(NULL, NULL);
    GMainLoop *loop = g_main_loop_new(NULL, FALSE);

    // 打印GStreamer版本信息
    std::cout << "using GStreamer version: " << gst_version_string() << std::endl;

    // 打开输入流
    cv::VideoCapture cap(input_rtmp);
    if (!cap.isOpened()) {
        std::cerr << "cannot open input source: " << input_rtmp << std::endl;
        return -1;
    }
    
    // 打印输入源信息
    if (input_rtmp.find("rtmp://") == 0) {
        std::cout << "reading RTMP stream" << std::endl;
    } else if (input_rtmp.find("rtsp://") == 0) {
        std::cout << "reading RTSP stream" << std::endl;
    } else {
        std::cout << "reading local file or device" << std::endl;
    }

    double input_fps = cap.get(cv::CAP_PROP_FPS);
    if (input_fps <= 0 || input_fps > 120) {
        std::cout << "warning: abnormal FPS value (" << input_fps << "), set to default value 30" << std::endl;
        input_fps = 30.0;
    }
    
    double frame_duration = 1.0 / input_fps * GST_SECOND;
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    std::cout << "input video: " << width << "x" << height << " @ " << input_fps << "fps" << std::endl;

    // 读取第一帧
    cv::Mat first_frame;
    if (!cap.read(first_frame)) {
        std::cerr << "cannot read first frame" << std::endl;
        return -1;
    }

    // 构建GStreamer流媒体管道 - 使用H264编码输出
    GstElement *pipeline = gst_pipeline_new("video-pipeline");
    GstElement *source = gst_element_factory_make("appsrc", "mysource");
    GstElement *videoconvert = gst_element_factory_make("videoconvert", "videoconvert");
    GstElement *x264enc = gst_element_factory_make("x264enc", "x264enc");
    GstElement *h264parse = gst_element_factory_make("h264parse", "h264parse");
    GstElement *flvmux = gst_element_factory_make("flvmux", "flvmux");
    GstElement *queue = gst_element_factory_make("queue", "queue");
    GstElement *rtmpsink = gst_element_factory_make("rtmpsink", "rtmpsink");

    // 检查元素是否成功创建
    if (!pipeline || !source || !videoconvert || !x264enc || !h264parse || 
        !flvmux || !queue || !rtmpsink) {
        std::cerr << "one or more elements cannot be created\n";
        if (!pipeline) std::cerr << "failed to create pipeline\n";
        if (!source) std::cerr << "failed to create appsrc\n";
        if (!videoconvert) std::cerr << "failed to create videoconvert\n";
        if (!x264enc) std::cerr << "failed to create x264enc\n";
        if (!h264parse) std::cerr << "failed to create h264parse\n";
        if (!flvmux) std::cerr << "failed to create flvmux\n";
        if (!queue) std::cerr << "failed to create queue\n";
        if (!rtmpsink) std::cerr << "创建rtmpsink失败\n";
        return -1;
    }

    // 设置元素属性
    g_object_set(G_OBJECT(rtmpsink), "location", output_rtmp.c_str(), NULL);
    
    // H264编码器参数优化
    g_object_set(G_OBJECT(x264enc),
        "tune", 0x00000004,        // zerolatency
        "speed-preset", 1,         // ultrafast
        "key-int-max", 30,         // 每30帧一个关键帧
        "bitrate", 2000,           // 2Mbps
        "threads", 4,              // 使用4线程
        NULL);

    // 队列配置
    g_object_set(G_OBJECT(queue),
        "max-size-buffers", 1000,
        "max-size-bytes", 0,
        "max-size-time", 0,
        "leaky", 2, // downstream
        NULL);

    // flvmux配置
    g_object_set(G_OBJECT(flvmux), 
        "streamable", TRUE,
        NULL);

    // 设置appsrc参数
    g_object_set(source, "caps",
                 gst_caps_new_simple("video/x-raw", 
                     "format", G_TYPE_STRING, "BGR",
                     "width", G_TYPE_INT, first_frame.cols,
                     "height", G_TYPE_INT, first_frame.rows,
                     "framerate", GST_TYPE_FRACTION, (int)input_fps, 1,
                     NULL),
                 "stream-type", 0,
                 "is-live", TRUE,
                 "format", GST_FORMAT_TIME,
                 NULL);

    // 设置总线监视器
    auto bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    // 将元素添加到管道
    gst_bin_add_many(GST_BIN(pipeline), 
        source, videoconvert, x264enc, h264parse, queue, flvmux, rtmpsink, NULL);
    
    // 链接管道元素 - 详细错误处理
    bool link_success = true;
    
    // 逐个链接元素以便于调试
    link_success &= gst_element_link(source, videoconvert);
    if (!link_success) {
        std::cerr << "failed to link: source -> videoconvert" << std::endl;
        return -1;
    }
    
    link_success &= gst_element_link(videoconvert, x264enc);
    if (!link_success) {
        std::cerr << "failed to link: videoconvert -> x264enc" << std::endl;
        return -1;
    }
    
    link_success &= gst_element_link(x264enc, h264parse);
    if (!link_success) {
        std::cerr << "failed to link: x264enc -> h264parse" << std::endl;
        return -1;
    }
    
    link_success &= gst_element_link(h264parse, queue);
    if (!link_success) {
        std::cerr << "failed to link: h264parse -> queue" << std::endl;
        return -1;
    }
    
    link_success &= gst_element_link(queue, flvmux);
    if (!link_success) {
        std::cerr << "failed to link: queue -> flvmux" << std::endl;
        return -1;
    }
    
    link_success &= gst_element_link(flvmux, rtmpsink);
    if (!link_success) {
        std::cerr << "failed to link: flvmux -> rtmpsink" << std::endl;
        return -1;
    }

    std::cout << "GStreamer pipeline created successfully, starting media stream..." << std::endl;

    // 启动管道
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr("failed to set pipeline to playing state.\n");
        gst_object_unref(pipeline);
        return -1;
    }

    // 初始化YOLO检测器
    YoloDetecter detecter(modelPath);
    
    // 初始化ByteTrack跟踪器
    BYTETracker tracker(input_fps, trackBuffer, track_thresh, high_thresh, match_thresh);

    // 计数线的X坐标（图像中心的垂直线）
    int line_x = width / 2;

    // 重置计数器和轨迹数据
    counter = 0;
    counted_tracks.clear();
    trajectories.clear();

    // 启动读取和处理帧的线程
    std::thread read_thread(read_frames, std::ref(cap));
    std::thread process_thread(process_frames, std::ref(detecter), std::ref(tracker), line_x, show_gui);

    GstClockTime last_pts = 0;
    int frames_since_last_pts = 0;

    // 主循环：从输出队列获取处理后的帧并推送到GStreamer管道
    while (is_running) {
        FrameData frame_data;
        {
            std::unique_lock<std::mutex> lock(output_mutex);
            output_cv.wait(lock, [] { return !output_queue.empty() || !is_running; });
            if (!is_running && output_queue.empty()) break;
            frame_data = output_queue.front();
            output_queue.pop();
        }

        // 创建GstBuffer并复制帧数据
        GstBuffer *buffer = gst_buffer_new_allocate(nullptr, 
            frame_data.frame.total() * frame_data.frame.elemSize(), nullptr);
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);
        std::memcpy(map.data, frame_data.frame.data, map.size);
        gst_buffer_unmap(buffer, &map);

        // 设置缓冲区时间戳
        GstClockTime current_pts = frame_data.timestamp;
        if (current_pts <= last_pts) {
            current_pts = last_pts + frame_duration * (frames_since_last_pts + 1);
            frames_since_last_pts++;
        } else {
            last_pts = current_pts;
            frames_since_last_pts = 0;
        }

        GST_BUFFER_PTS(buffer) = current_pts;
        GST_BUFFER_DURATION(buffer) = frame_duration;

        // 将缓冲区推送到appsrc
        GstFlowReturn push_ret;
        g_signal_emit_by_name(source, "push-buffer", buffer, &push_ret);
        gst_buffer_unref(buffer);

        if (push_ret != GST_FLOW_OK) {
            std::cerr << "error pushing buffer to pipeline" << std::endl;
            break;
        }
    }

    // 清理并退出
    is_running = false;
    input_cv.notify_all();
    output_cv.notify_all();
    read_thread.join();
    process_thread.join();

    gst_element_send_event(source, gst_event_new_eos());
    g_main_loop_run(loop);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_main_loop_unref(loop);
    cap.release();
    
    return 0;
}

// 添加读取配置文件的函数
bool readConfigFile(const std::string& filename, 
                   std::string& input_rtmp, 
                   std::string& output_rtmp, 
                   std::string& model_path, 
                   int& track_buffer_size, 
                   bool& show_gui,
                   float& track_thresh,
                   float& high_thresh,
                   float& match_thresh) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "cannot open config file: " << filename << std::endl;
        return false;
    }

    std::string line;
    std::string section;

    // 默认值
    track_buffer_size = 120;
    show_gui = false;
    track_thresh = 0.5;
    high_thresh = 0.6;
    match_thresh = 0.4;

    while (std::getline(file, line)) {
        // 移除前导和尾随空白
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        // 跳过空行和注释
        if (line.empty() || line[0] == ';' || line[0] == '#')
            continue;

        // 检查是否为节定义 [Section]
        if (line[0] == '[' && line[line.size() - 1] == ']') {
            section = line.substr(1, line.size() - 2);
            continue;
        }

        // 解析键值对
        size_t delimiter_pos = line.find('=');
        if (delimiter_pos != std::string::npos) {
            std::string key = line.substr(0, delimiter_pos);
            std::string value = line.substr(delimiter_pos + 1);
            
            // 移除键和值的前导/尾随空白
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            // 根据节和键设置值
            if (section == "Stream") {
                if (key == "input_rtmp") input_rtmp = value;
                else if (key == "output_rtmp") output_rtmp = value;
            } else if (section == "Model") {
                if (key == "model_path") model_path = value;
            } else if (section == "Tracking") {
                if (key == "track_buffer_size") {
                    try {
                        track_buffer_size = std::stoi(value);
                    } catch (const std::exception& e) {
                        std::cerr << "invalid track_buffer_size value, using default value 120" << std::endl;
                        track_buffer_size = 120;
                    }
                } else if (key == "show_gui") {
                    show_gui = (value == "true" || value == "1" || value == "yes");
                } else if (key == "track_thresh") {
                    try {
                        track_thresh = std::stof(value);
                    } catch (const std::exception& e) {
                        std::cerr << "invalid track_thresh value, using default value 0.5" << std::endl;
                    }
                } else if (key == "high_thresh") {
                    try {
                        high_thresh = std::stof(value);
                    } catch (const std::exception& e) {
                        std::cerr << "invalid high_thresh value, using default value 0.6" << std::endl;
                    }
                } else if (key == "match_thresh") {
                    try {
                        match_thresh = std::stof(value);
                    } catch (const std::exception& e) {
                        std::cerr << "invalid match_thresh value, using default value 0.4" << std::endl;
                    }
                }
            }
        }
    }

    // 检查是否设置了必需参数
    if (input_rtmp.empty() || output_rtmp.empty() || model_path.empty()) {
        std::cerr << "config file missing required parameters" << std::endl;
        if (input_rtmp.empty()) std::cerr << "missing: input_rtmp" << std::endl;
        if (output_rtmp.empty()) std::cerr << "missing: output_rtmp" << std::endl;
        if (model_path.empty()) std::cerr << "missing: model_path" << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char *argv[])
{
    // 设置本地化支持（移到这里）
    std::locale::global(std::locale(""));
    std::cout.imbue(std::locale(""));
    
    std::string input_rtmp;
    std::string output_rtmp;
    std::string model_path;
    int track_buffer_size = 120;
    bool show_gui = false;
    float track_thresh = 0.5;
    float high_thresh = 0.6;
    float match_thresh = 0.4;

    // 首先检查是否通过命令行指定配置文件
    if (argc == 2) {
        // 只有一个参数，假设是配置文件路径
        std::string config_file = argv[1];
        if (!readConfigFile(config_file, input_rtmp, output_rtmp, model_path, track_buffer_size, show_gui, track_thresh, high_thresh, match_thresh)) {
            return -1;
        }
        std::cout << "loading settings from config file: " << config_file << std::endl;
    } 
    // 原始命令行参数处理
    else if (argc >= 4 && argc <= 6) {
        input_rtmp = argv[1];
        output_rtmp = argv[2];
        model_path = argv[3];
        
        // 可选参数: 跟踪缓冲区大小
        if (argc >= 5) {
            try {
                track_buffer_size = std::stoi(argv[4]);
                std::cout << "using custom track buffer size: " << track_buffer_size << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "invalid track buffer size parameter, using default value 120" << std::endl;
            }
        }
        
        // 可选参数: 显示GUI
        if (argc >= 6) {
            std::string gui_param = argv[5];
            show_gui = (gui_param == "true" || gui_param == "1");
            std::cout << "GUI display: " << (show_gui ? "enabled" : "disabled") << std::endl;
        }
    } 
    // 尝试加载默认配置文件
    else if (argc == 1) {
        const std::string default_config = "config.ini";
        if (std::ifstream(default_config).good()) {
            if (!readConfigFile(default_config, input_rtmp, output_rtmp, model_path, track_buffer_size, show_gui, track_thresh, high_thresh, match_thresh)) {
                std::cerr << "failed to load settings from default config file" << std::endl;
                
                std::cerr << "invalid parameters" << std::endl;
                std::cerr << "usage: ./main [config.ini]" << std::endl;
                std::cerr << "    or: ./main [input_rtmp] [output_rtmp] [model_path] [optional:track_buffer_size] [optional:show_gui]" << std::endl;
                std::cerr << "example: " << std::endl;
                std::cerr << "  use config file: ./main config.ini" << std::endl;
                std::cerr << "  local file: ./main ./test_videos/demo.mp4 rtmp://127.0.0.1:1935/live/output ../yolo/engine/yolov8s.engine" << std::endl;
                return -1;
            }
            std::cout << "loading settings from default config file" << std::endl;
        } else {
            std::cerr << "invalid parameters" << std::endl;
            std::cerr << "usage: ./main [config.ini]" << std::endl;
            std::cerr << "    or: ./main [input_rtmp] [output_rtmp] [model_path] [optional:track_buffer_size] [optional:show_gui]" << std::endl;
            std::cerr << "example: " << std::endl;
            std::cerr << "  use config file: ./main config.ini" << std::endl;
            std::cerr << "  local file: ./main ./test_videos/demo.mp4 rtmp://127.0.0.1:1935/live/output ../yolo/engine/yolov8s.engine" << std::endl;
            return -1;
        }
    } else {
        std::cerr << "invalid parameters" << std::endl;
        std::cerr << "usage: ./main [config.ini]" << std::endl;
        std::cerr << "    or: ./main [input_rtmp] [output_rtmp] [model_path] [optional:track_buffer_size] [optional:show_gui]" << std::endl;
        std::cerr << "example: " << std::endl;
        std::cerr << "  use config file: ./main config.ini" << std::endl;
        std::cerr << "  local file: ./main ./test_videos/demo.mp4 rtmp://127.0.0.1:1935/live/output ../yolo/engine/yolov8s.engine" << std::endl;
        return -1;
    }

    // 打印加载的配置
    std::cout << "using the following settings:" << std::endl;
    std::cout << "  input source: " << input_rtmp << std::endl;
    std::cout << "  output stream: " << output_rtmp << std::endl;
    std::cout << "  model path: " << model_path << std::endl;
    std::cout << "  track buffer size: " << track_buffer_size << std::endl;
    std::cout << "  GUI display: " << (show_gui ? "enabled" : "disabled") << std::endl;
    std::cout << "  track thresh: " << track_thresh << std::endl;
    std::cout << "  high thresh: " << high_thresh << std::endl;
    std::cout << "  match thresh: " << match_thresh << std::endl;

    return run(input_rtmp, output_rtmp, model_path, track_buffer_size, show_gui,
              track_thresh, high_thresh, match_thresh);
}
