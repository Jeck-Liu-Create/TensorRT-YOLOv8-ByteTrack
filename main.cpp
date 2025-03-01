#include <iostream>
#include <string>
#include "yolov8_lib.h"
#include "BYTETracker.h"
#include <thread>
#include <queue>
#include <map>
#include <deque>
#include <set>

// 用于存储每个轨迹的中心点历史
std::map<int, std::deque<cv::Point>> trajectories;
// 用于存储已经计数的跟踪ID
std::set<int> counted_tracks;
// 计数器
int counter = 0;

std::vector<int> trackClasses{0};  // Steel coil

bool isTrackingClass(int class_id) {
    for (auto& c : trackClasses) {
        if (class_id == c) return true;
    }
    return false;
}

// 在图像上绘制垂直虚线和计数信息
void drawCountingLine(cv::Mat& img, int count) {
    // 绘制垂直虚线 (在图像中间垂直位置)
    int line_x = img.cols / 2;
    cv::Scalar lineColor(0, 255, 0); // 绿色虚线
    
    // 绘制垂直虚线效果
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
            // 使用跟踪ID对应的颜色绘制轨迹线
            cv::line(img, points[i-1], points[i], 
                    tracker.get_color(track.first), 2);
        }
    }
}

// 更新轨迹并检查计数
void updateTrajectories(const std::vector<STrack>& tracks, int line_x) {
    for (const auto& track : tracks) {
        int track_id = track.track_id;
        
        // 获取当前物体的中心点
        std::vector<float> tlwh = track.tlwh;
        int x_center = tlwh[0] + tlwh[2]/2;
        int y_center = tlwh[1] + tlwh[3]/2;
        cv::Point center_point(x_center, y_center);
        
        // 更新轨迹
        if (trajectories.find(track_id) == trajectories.end()) {
            trajectories[track_id] = std::deque<cv::Point>();
        }
        trajectories[track_id].push_back(center_point);
        
        // 限制轨迹长度，防止内存占用过大
        if (trajectories[track_id].size() > 30) {
            trajectories[track_id].pop_front();
        }
        
        // 检查是否穿过计数线
        if (trajectories[track_id].size() >= 2) {
            auto& points = trajectories[track_id];
            cv::Point prev = points[points.size() - 2];
            cv::Point curr = points[points.size() - 1];
            
            // 检查是否穿过了垂直线 (从右到左或从左到右)
            if ((prev.x > line_x && curr.x <= line_x) || 
                (prev.x <= line_x && curr.x > line_x)) {
                // 如果这个ID还没被计数过
                if (counted_tracks.find(track_id) == counted_tracks.end()) {
                    counter++; // 增加计数
                    counted_tracks.insert(track_id); // 标记为已计数
                }
            }
        }
    }
}

int run(const std::string& videoPath, const std::string& modelPath, int trackBuffer)
{
    bool isRTMP = videoPath.find("rtmp://") == 0;
    
    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_BUFFERSIZE, 3); // 设置缓冲区大小
    
    int retry_count = 0;
    const int max_retry = 5;
    
    // 增加视频流读取的容错处理
    while (retry_count < max_retry) {
        if (isRTMP) {
            std::cout << "Reading RTMP stream: " << videoPath << std::endl;
            // 使用通用硬件加速，指定第一个GPU设备
            cap.open(videoPath, cv::CAP_FFMPEG, {
                cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY,
                cv::CAP_PROP_HW_DEVICE, 0  // 指定第一个GPU设备
            });
        } else {
            std::cout << "Reading local video file: " << videoPath << std::endl;
            cap.open(videoPath);
        }
        
        if (cap.isOpened()) break;
        
        std::cerr << "Unable to open video source, retrying..." << std::endl;
        retry_count++;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    if (!cap.isOpened()) {
        std::cerr << "Failed to open video source after " << max_retry << " attempts. Ensure FFmpeg supports H265 and CUDA." << std::endl;
        return -1;
    }

    // 检查硬件加速状态
    int acceleration = cap.get(cv::CAP_PROP_HW_ACCELERATION);
    std::cout << "Hardware acceleration: " << (acceleration > 0 ? "Enabled" : "None") << std::endl;

    // 获取视频基本信息
    int img_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int img_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Total frames: " << nFrame << std::endl;

    // 初始化YOLO检测器
    std::string trtFile = modelPath; 
    YoloDetecter detecter(trtFile);
    
    // 初始化ByteTrack跟踪器，使用传入的trackBuffer参数
    BYTETracker tracker(fps, trackBuffer);

    // 计数线的x坐标（在图像中央垂直方向）
    int line_x = img_w / 2;

    cv::Mat img;
    int num_frames = 0;
    int total_ms = 0;
    const int QUEUE_SIZE = 3;
    std::queue<cv::Mat> frame_queue;
    
    // 重置计数器和轨迹数据
    counter = 0;
    counted_tracks.clear();
    trajectories.clear();

    while (true)
    {
        if (!cap.read(img)) break;
        num_frames++;
        
        // 检查帧完整性
        if (img.empty() || img.cols <= 0 || img.rows <= 0) {
            std::cerr << "Frame " << num_frames << " is invalid or empty!" << std::endl;
            continue;
        }

        // 添加到队列
        cv::Mat temp;
        img.copyTo(temp);
        frame_queue.push(temp);
        
        // 保持队列大小
        if (frame_queue.size() > QUEUE_SIZE) {
            frame_queue.front().release();
            frame_queue.pop();
        }
        
        // 使用当前帧（禁用平滑以避免模糊）
        cv::Mat display_img = img.clone();  // 可选：取消平滑测试效果

        // 执行YOLO推理
        auto start = std::chrono::system_clock::now();
        std::vector<DetectResult> res = detecter.inference(display_img);

        // 过滤需要跟踪的类别
        std::vector<Object> objects;
        for (long unsigned int j = 0; j < res.size(); j++)
        {
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
        for (int i = 0; i < output_stracks.size(); i++)
        {
            std::vector<float> tlwh = output_stracks[i].tlwh;
            if (tlwh[2] * tlwh[3] > 20)
            {
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
        cv::putText(display_img, cv::format("frame: %d fps: %d num: %ld", num_frames, num_frames * 1000000 / total_ms, output_stracks.size()), 
                    cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        // 创建可调整大小的窗口并显示（避免缩放模糊）
        cv::namedWindow("img", cv::WINDOW_NORMAL);
        cv::imshow("img", display_img); 
        int c = cv::waitKey(1);
        if (c == 27) break;

        // 释放内存
        display_img.release();
        img.release();
    }

    // 释放资源
    cap.release();
    std::cout << "FPS: " << num_frames * 1000000 / total_ms << std::endl;

    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4) {
        std::cerr << "Invalid parameters!" << std::endl;
        std::cerr << "Usage: ./main [video_path_or_RTMP_URL] [model_path] [optional:track_buffer_size]" << std::endl;
        std::cerr << "Examples: " << std::endl;
        std::cerr << "  Local file: ./main ./test_videos/demo.mp4 ../yolo/engine/yolov8s.engine" << std::endl;
        std::cerr << "  RTMP stream: ./main rtmp://your_rtmp_server/live/stream ../yolo/engine/yolov8s.engine" << std::endl;
        std::cerr << "  Custom track buffer: ./main ./test_videos/demo.mp4 ../yolo/engine/yolov8s.engine 240" << std::endl;
        return -1;
    }

    std::string videoPath = argv[1];
    std::string modelPath = argv[2];
    
    // Default track buffer size is 120, can be modified via third parameter
    int trackBuffer = 120;
    if (argc == 4) {
        try {
            trackBuffer = std::stoi(argv[3]);
            std::cout << "Using custom track buffer size: " << trackBuffer << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Invalid track buffer parameter, using default value 120" << std::endl;
        }
    }

    return run(videoPath, modelPath, trackBuffer);
}