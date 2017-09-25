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

void capture(INectaCamera &cam) {
    // Get list of connected cameras
    Array<String> camList = cam.GetCameraList();
    if (camList.Size() == 0) {
        cerr << "No AlkUSB3 camera found" << endl;
        exit(1);
    }

    cam.SetCamera(0);
    cam.Init();

    cam.SetImageSizeX(cols);
    cam.SetImageSizeY(rows);

    cerr  << "Starting acquisition..." << endl;
    // Start acquisition

    int nOfSamples = 10000;

    cv::Mat cv_image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int f = 0; true; f++) {
        if (cv::getWindowProperty("necta", 0) < 0) {
            cout
                << cv::getTrackbarPos("analog gain", "necta") << " "
                << cv::getTrackbarPos("cds gain", "necta") << " "
                << cv::getTrackbarPos("shutterspeed", "necta") << " "
                << cv::getTrackbarPos("gamma", "necta") / 1000.0 << endl;
            break;
        }
        cam.SetGain(cv::getTrackbarPos("analog gain", "necta"));
        cam.SetCDSGain(cv::getTrackbarPos("cds gain", "necta"));
        cam.SetShutter(cv::getTrackbarPos("shutterspeed", "necta"));
        cam.SetGamma(cv::getTrackbarPos("gamma", "necta") / 1000.0);
        cam.SetAcquire(true);
        // Get new frame
        stringstream ss;
        ss << "im" << setfill('0') << setw(4) << f << ".bmp";
        auto raw_image = cam.GetImageData();
        cam.SetAcquire(false);
        cv_image.data = raw_image.Data();
        cv::imshow("necta", cv_image.clone());
        // cam.GetImagePtr(true).Save(ss.str().c_str());
        cv::waitKey(1);
    }

    cerr << "Acquisition stopped" << endl;
}

int main(int argc, char *argv[]) {
    INectaCamera &cam = INectaCamera::Create();
    cv::namedWindow("necta", cv::WINDOW_NORMAL);

    int analog_gain = 10;
    int cds_gain = 1;
    int shutterspeed = 2000;
    int gamma = 1000;

    cv::createTrackbar("shutterspeed", "necta", &shutterspeed, 10000);
    cv::createTrackbar("analog gain", "necta", &analog_gain, 20);
    cv::createTrackbar("cds gain", "necta", &cds_gain, 1);
    cv::createTrackbar("gamma", "necta", &gamma, 2000);

    // capture
    try {
        capture(cam);
    } catch (const Exception &ex) {
        cerr << "Exception " << ex.Name() << " occurred" << endl
            << ex.Message() << endl;
    }
    cout
        << cv::getTrackbarPos("analog gain", "necta") << " "
        << cv::getTrackbarPos("cds gain", "necta") << " "
        << cv::getTrackbarPos("shutterspeed", "necta") << " "
        << cv::getTrackbarPos("gamma", "necta") / 1000.0 << endl;
    return 0;
}
