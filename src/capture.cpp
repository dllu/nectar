#include <INectaCamera.h>
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace CAlkUSB3;
using namespace std;

const int rows = 2048;
const int cols = 4096;

void capture(INectaCamera &cam, int shutterspeed, int analog_gain, int cds_gain, int gamma) {
    // Get list of connected cameras
    Array<String> camList = cam.GetCameraList();
    if (camList.Size() == 0) {
        cout << "No AlkUSB3 camera found" << endl;
        exit(1);
    }

    cam.SetCamera(0);
    cam.Init();

    cam.SetImageSizeX(cols);
    cam.SetImageSizeY(rows);
    cam.SetGain(analog_gain);
    cam.SetCDSGain(cds_gain);
    cam.SetShutter(shutterspeed);
    cam.SetGamma(gamma / 1000.0);

    cout << "Starting acquisition..." << endl;
    // Start acquisition

    int nOfSamples = 300;

    cam.SetAcquire(true);
    for (int f = 0; f < nOfSamples; f++) {
        // Get new frame
        stringstream ss;
        ss << "im" << setfill('0') << setw(4) << f << ".bmp";
        cam.GetImagePtr(true).Save(ss.str().c_str());
    }

    // Stop acquisition
    cam.SetAcquire(false);
    cout << "Acquisition stopped" << endl;
}

int main(int argc, char *argv[]) {
    INectaCamera &cam = INectaCamera::Create();
    if (argc < 5) {
        cerr << "USAGE: capture shutterspeed analog_gain cds_gain gamma" << endl;
        return 1;
    }

    int analog_gain = atoi(argv[1]);
    int cds_gain = atoi(argv[2]);
    int shutterspeed = atoi(argv[3]);
    int gamma = atoi(argv[4]);

    // capture
    try {
        capture(cam, shutterspeed, analog_gain, cds_gain, gamma);
    } catch (const Exception &ex) {
        cout << "Exception " << ex.Name() << " occurred" << endl
            << ex.Message() << endl;
    }
    return 0;
}
