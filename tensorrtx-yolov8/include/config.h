#define USE_FP16
//#define USE_INT8

const static char *kInputTensorName = "images";
const static char *kOutputTensorName = "output";
const static int kNumClass = 1;
const static int kBatchSize = 1;
const static int kGpuId = 0;
const static int kInputH = 800;
const static int kInputW = 800;
const static float kNmsThresh = 0.3f;
const static float kConfThresh = 0.3f;
const static int kMaxInputImageSize = 3000 * 3000;
const static int kMaxNumOutputBbox = 1000;
