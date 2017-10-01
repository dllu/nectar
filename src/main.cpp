#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <INectaCamera.h>
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace CAlkUSB3;
using namespace std;

const int rows = 192;
const int cols = 4096;

void printColorCodings(INectaCamera &cam) {
    auto acc = cam.GetAvailableColorCoding();
    for (auto cc = acc.Front(); 1; cc++) {
        cerr << cc->ToString() << endl;
        if (cc == acc.Back()) {
            break;
        }
    }
}

void printVideoModes(INectaCamera &cam) {
    auto acc = cam.GetAvailableVideoModes();
    for (auto cc = 0; cc < acc.Size(); cc++) {
        cerr << (int)(acc[cc].GetVideoMode()) << endl;
    }
    cerr << "selected: " << (int) cam.GetVideoMode() << endl;
}

void capture(INectaCamera &cam, int *gain, int *cds_gain, int *shutter) {
    // Get list of connected cameras
    Array<String> camList = cam.GetCameraList();
    if (camList.Size() == 0) {
        cerr << "No AlkUSB3 camera found" << endl;
        exit(1);
    }

    cam.SetCamera(0);
    cam.Init();

    cam.SetADCResolution(12);
    cam.SetVideoMode(1);
    cam.SetColorCoding(ColorCoding::Raw16);
    cam.SetImageSizeX(cols);
    cam.SetImageSizeY(rows);

    cam.SetPacketSize(min(16384U, cam.GetMaxPacketSize()));

    cerr  << "Starting acquisition..." << endl;
    // Start acquisition


    cam.SetAcquire(true);
    cv::Mat cv_image(rows, cols / 2, CV_8UC3, cv::Scalar(0, 0, 0));

    std::vector<int>reds(256, 0);
    std::vector<int>greens(256, 0);
    std::vector<int>blues(256, 0);
    for (int f = 0; true; f++) {
        if (cv::getWindowProperty("necta", 0) < 0) {
            break;
        }
        std::fill(reds.begin(), reds.end(), 0);
        std::fill(greens.begin(), greens.end(), 0);
        std::fill(blues.begin(), blues.end(), 0);
        cam.SetGain(*gain);
        cam.SetCDSGain(*cds_gain);
        cam.SetShutter(*shutter);

        // Get new frame
        auto raw_image = cam.GetRawData();
        for (int i = 0; i < rows/2; i++) {
            for (int j = 0; j < cols/2; j++) {
                cv::Vec3b &intensity = cv_image.at<cv::Vec3b>(i, j);

                intensity.val[0] = raw_image[2 * (2 * i * cols + (2 * j + 1)) + 1];
                intensity.val[1] = (raw_image[2 * ((2 * i + 1) * cols + (2 * j + 1)) + 1] + raw_image[2 * (2 * i * cols + 2 * j) + 1]) / 2;
                intensity.val[2] = raw_image[2 * ((2 * i + 1) * cols + 2 * j) + 1];

                reds[intensity.val[0]]++;
                greens[intensity.val[1]]++;
                blues[intensity.val[2]]++;
            }
        }
        for (int i = 0; i < rows/2; i++) {
            for (int j = 0; j < 256; j++) {
                cv::Vec3b &intensity = cv_image.at<cv::Vec3b>(rows - i - 1, j);
                if (i * 100 < reds[j]) {
                    intensity.val[0] = 255;
                } else {
                    intensity.val[0] = 0;
                }
                if (i * 100 < greens[j]) {
                    intensity.val[1] = 255;
                } else {
                    intensity.val[1] = 0;
                }
                if (i * 100 < blues[j]) {
                    intensity.val[2] = 255;
                } else {
                    intensity.val[2] = 0;
                }
            }
        }
        if (cv::getWindowProperty("necta", 0) < 0) {
            break;
        }
        cv::imshow("necta", cv_image);

        // stringstream ss;
        // ss << "im" << setfill('0') << setw(4) << f << ".bmp";
        // cam.GetImagePtr(true).Save(ss.str().c_str());
        if (cv::getWindowProperty("necta", 0) < 0) {
            break;
        }
        cv::waitKey(1);
    }
    cam.SetAcquire(false);

    cerr << "Acquisition stopped" << endl;
}

int main(int argc, char *argv[]) {
    INectaCamera &cam = INectaCamera::Create();
    cv::namedWindow("necta", cv::WINDOW_NORMAL);

    int analog_gain = 10;
    int cds_gain = 1;
    int shutterspeed = 2000;

    cv::createTrackbar("shutterspeed", "necta", &shutterspeed, 10000);
    cv::createTrackbar("analog gain", "necta", &analog_gain, 20);
    cv::createTrackbar("cds gain", "necta", &cds_gain, 1);

    // capture
    try {
        capture(cam, &analog_gain, &cds_gain, &shutterspeed);
    } catch (const Exception &ex) {
        cerr << "Exception " << ex.Name() << " occurred" << endl
            << ex.Message() << endl;
    } catch(...) {
        cerr << "Unhandled excpetion" << endl;
    }
    cout << analog_gain << " " << cds_gain << " " << shutterspeed << endl;
    return 0;
}
