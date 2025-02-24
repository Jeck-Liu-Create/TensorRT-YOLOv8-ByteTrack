#include <iostream>
#include <string>
#include "yolov8_lib.h"
#include "BYTETracker.h"
#include <thread>

std::vector<int>  trackClasses {0};  // Steel coil

bool isTrackingClass(int class_id){
	for (auto& c : trackClasses){
		if (class_id == c) return true;
	}
	return false;
}


int run(const std::string& videoPath, const std::string& modelPath)
{
    bool isRTMP = videoPath.find("rtmp://") == 0;
    
    cv::VideoCapture cap;
    int retry_count = 0;
    const int max_retry = 5;
    
    // 增加视频流读取的容错处理
    while (retry_count < max_retry) {
        if (isRTMP) {
            std::cout << "Reading RTMP stream: " << videoPath << std::endl;
            cap.open(videoPath, cv::CAP_FFMPEG);
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
        std::cerr << "Failed to open video source after " << max_retry << " attempts" << std::endl;
        return -1;
    }

    // 启用GPU加速视频解码
    cap.set(cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY);

    // 获取视频基本信息
    int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
    int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << endl;

    // 初始化视频写入器
    cv::VideoWriter writer("result.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));

    // 初始化YOLO检测器
    std::string trtFile = modelPath; 
    YoloDetecter detecter(trtFile);
    
    // 初始化ByteTrack跟踪器
    BYTETracker tracker(fps, 30);

    cv::Mat img;
    int num_frames = 0;
    int total_ms = 0;
    while (true)
    {
        if(!cap.read(img)) break;
        num_frames ++;
        
        // 每2帧处理一次，降低处理频率
        if (num_frames % 1 != 0) continue;
        
        // 每20帧打印一次处理进度
        if (num_frames % 20 == 0)
        {
            cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << endl;
        }
        if (img.empty()) break;

        auto start = std::chrono::system_clock::now();
        
        // 执行YOLO推理
        std::vector<DetectResult> res = detecter.inference(img);

        // 过滤需要跟踪的类别
        std::vector<Object> objects;
        for (long unsigned int j = 0; j < res.size(); j++)
        {
            cv::Rect r = res[j].tlwh;
            float conf = (float)res[j].conf;
            int class_id = (int)res[j].class_id;

            if (isTrackingClass(class_id)){
                cv::Rect_<float> rect((float)r.x, (float)r.y, (float)r.width, (float)r.height);
                Object obj {rect, class_id, conf};
                objects.push_back(obj);
            }
        }

        // 执行目标跟踪
        std::vector<STrack> output_stracks = tracker.update(objects);

        auto end = std::chrono::system_clock::now();
        total_ms = total_ms + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // 绘制跟踪结果
        for (int i = 0; i < output_stracks.size(); i++)
        {
            std::vector<float> tlwh = output_stracks[i].tlwh;
            if (tlwh[2] * tlwh[3] > 20)
            {
                cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
                cv::putText(img, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5), 
                        0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::rectangle(img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
            }
        }
        // 显示帧率和目标数量
        cv::putText(img, cv::format("frame: %d fps: %d num: %ld", num_frames, num_frames * 1000000 / total_ms, output_stracks.size()), 
                cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        writer.write(img);

        // 显示图像
        cv::imshow("img", img); 
        int c = cv::waitKey(1);
        if (c == 27) break;  // ESC键退出
    }

    // 释放资源
    cap.release();
    std::cout << "FPS: " << num_frames * 1000000 / total_ms << std::endl;

    return 0;
}


int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cerr << "Invalid arguments!" << std::endl;
        std::cerr << "Usage: ./main [video_path_or_RTMP_URL] [model_path]" << std::endl;
        std::cerr << "Examples: " << std::endl;
        std::cerr << "  Local file: ./main ./test_videos/demo.mp4 ../yolo/engine/yolov8s.engine" << std::endl;
        std::cerr << "  RTMP stream: ./main rtmp://your_rtmp_server/live/stream ../yolo/engine/yolov8s.engine" << std::endl;
        return -1;
    }

    return run(argv[1], argv[2]);
}
