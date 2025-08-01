#include "yolo/postprocess.h"
#include "yolo/utils.h"
#include "yolo/config.h"

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    float l, r, t, b;
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);

    if (r_h > r_w) {
        l = bbox[0];
        r = bbox[2];
        t = bbox[1] - (kInputH - r_w * img.rows) / 2;
        b = bbox[3] - (kInputH - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - (kInputW - r_h * img.cols) / 2;
        r = bbox[2] - (kInputW - r_h * img.cols) / 2;
        t = bbox[1];
        b = bbox[3];
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    l = std::max(0.0f, l);
    t = std::max(0.0f, t);
    int width = std::max(0, std::min(int(round(r - l)), img.cols - int(round(l))));
    int height = std::max(0, std::min(int(round(b - t)), img.rows - int(round(t))));

    return cv::Rect(int(round(l)), int(round(t)), width, height);
}

static float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            (std::max)(lbox[0], rbox[0]),
            (std::min)(lbox[2], rbox[2]),
            (std::max)(lbox[1], rbox[1]),
            (std::min)(lbox[3], rbox[3]),
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    float unionBoxS = (lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS;
    return interBoxS / unionBoxS;
}

static bool cmp(const Detection& a, const Detection& b) {
    if (a.conf == b.conf) {
        return a.bbox[0] < b.bbox[0];
    }
    return a.conf > b.conf;
}

void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;

    for (int i = 0; i < output[0]; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh || isnan(output[1 + det_size * i + 4]))
            continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0)
            m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

void batch_nms(std::vector<std::vector<Detection>>& res_batch, float* output, int batch_size, int output_size,
               float conf_thresh, float nms_thresh) {
    res_batch.resize(batch_size);
    for (int i = 0; i < batch_size; i++) {
        nms(res_batch[i], &output[i * output_size], conf_thresh, nms_thresh);
    }
}

static cv::Mat scale_mask(cv::Mat mask, cv::Mat img) {
    int w, h, x, y;
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    if (r_h > r_w) {
        w = kInputW;
        h = r_w * img.rows;
        x = 0;
        y = (kInputH - h) / 2;
    } else {
        w = r_h * img.cols;
        h = kInputH;
        x = (kInputW - w) / 2;
        y = 0;
    }
    cv::Rect r(x, y, w, h);
    cv::Mat res;
    cv::resize(mask(r), res, img.size());
    return res;
}

void draw_mask_bbox(cv::Mat& img, std::vector<Detection>& dets, std::vector<cv::Mat>& masks,
                    std::unordered_map<int, std::string>& labels_map) {
    static std::vector<uint32_t> colors = {0xFF3838, 0xFF9D97, 0xFF701F, 0xFFB21D, 0xCFD231, 0x48F90A, 0x92CC17,
                                           0x3DDB86, 0x1A9334, 0x00D4BB, 0x2C99A8, 0x00C2FF, 0x344593, 0x6473FF,
                                           0x0018EC, 0x8438FF, 0x520085, 0xCB38FF, 0xFF95C8, 0xFF37C7};
    for (size_t i = 0; i < dets.size(); i++) {
        cv::Mat img_mask = scale_mask(masks[i], img);
        auto color = colors[(int)dets[i].class_id % colors.size()];
        auto bgr = cv::Scalar(color & 0xFF, color >> 8 & 0xFF, color >> 16 & 0xFF);

        cv::Rect r = get_rect(img, dets[i].bbox);
        for (int x = r.x; x < r.x + r.width; x++) {
            for (int y = r.y; y < r.y + r.height; y++) {
                float val = img_mask.at<float>(y, x);
                if (val <= 0.5)
                    continue;
                img.at<cv::Vec3b>(y, x)[0] = img.at<cv::Vec3b>(y, x)[0] / 2 + bgr[0] / 2;
                img.at<cv::Vec3b>(y, x)[1] = img.at<cv::Vec3b>(y, x)[1] / 2 + bgr[1] / 2;
                img.at<cv::Vec3b>(y, x)[2] = img.at<cv::Vec3b>(y, x)[2] / 2 + bgr[2] / 2;
            }
        }

        cv::rectangle(img, r, bgr, 2);

        // Get the size of the text
        cv::Size textSize =
                cv::getTextSize(labels_map[(int)dets[i].class_id] + " " + to_string_with_precision(dets[i].conf),
                                cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
        // Set the top left corner of the rectangle
        cv::Point topLeft(r.x, r.y - textSize.height);

        // Set the bottom right corner of the rectangle
        cv::Point bottomRight(r.x + textSize.width, r.y + textSize.height);

        // Draw the rectangle on the image
        cv::rectangle(img, topLeft, bottomRight, bgr, -1);

        cv::putText(img, labels_map[(int)dets[i].class_id] + " " + to_string_with_precision(dets[i].conf),
                    cv::Point(r.x, r.y + 4), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar::all(0xFF), 2);
    }
}
