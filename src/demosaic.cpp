#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "tinypng.h"

constexpr size_t rows = 2048;
constexpr size_t cols = 4096;
constexpr size_t start_cols = 100;
constexpr size_t match_px_top = 350;
constexpr size_t match_px_bot = 240;
constexpr double match_thresh = 0.02;

using ColVec = Eigen::Matrix<double, cols / 2, 1>;
using vecColVec = std::vector<ColVec, Eigen::aligned_allocator<ColVec>>;

constexpr size_t black_point_calib_start = 1294;
constexpr size_t black_point_calib_end = 1314;
constexpr double black_point_calib[] = {
    -185.60000, -194.40000, -28.00000,  -99.20000,  -405.60000, -370.00000,
    -353.60000, -402.40000, -182.40000, -152.80000, -427.20000, -439.20000,
    400.40000,  430.00000,  141.60000,  181.60000,  315.20000,  279.20000,
    354.80000,  415.20000,  -265.60000, -283.20000, -98.40000,  -166.40000,
    -238.80000, -285.60000, -192.80000, -286.40000, 183.20000,  255.20000,
    126.00000,  167.60000,  208.00000,  224.80000,  111.20000,  215.20000,
    -56.80000,  -128.80000, -88.80000,  -130.40000, -67.20000,  -118.40000,
    -69.60000,  -125.20000, 137.60000,  134.40000,  104.00000,  49.60000,
    266.40000,  335.20000,  74.40000,   144.00000,  504.00000,  434.00000,
    365.60000,  344.80000,  203.20000,  221.60000,  379.20000,  427.20000,
    -284.00000, -256.00000, -70.40000,  -72.00000,  -217.60000, -137.60000,
    -333.60000, -324.00000, 180.00000,  182.80000,  0.00000,    64.00000,
    110.80000,  143.20000,  59.20000,   144.80000,  -21.60000,  -132.00000,
    -21.60000,  -46.40000,
};

class PostProcessor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

   private:
    ColVec blue_start;
    ColVec red_start;
    ColVec green_start;

    ColVec blue_corrector;
    ColVec red_corrector;
    ColVec green_r_corrector;
    ColVec green_b_corrector;
    size_t start_n;

    /**
     * Each column may have a slightly different exposure due to electrical
     * noise in the sensor readout. Correct for this error by performing a
     * linear regression against the _start columns, which are averaged over
     * start_cols. The regression variables are intensity and y value (index
     * along the col).
     */
    static void matchColumnExposure(ColVec &mutable_col, const ColVec &target,
                                    const double thresh) {
        constexpr size_t acols = 3;
        Eigen::Matrix<double, cols / 2, acols> A;
        A.col(0) = mutable_col;
        A.col(1).setLinSpaced(0, 1);
        A.col(2).setOnes();

        Eigen::Matrix<double, match_px_bot + match_px_top, acols> AA;
        AA.block<match_px_bot, acols>(0, 0) =
            A.block<match_px_bot, acols>(0, 0);
        AA.block<match_px_top, acols>(match_px_bot, 0) =
            A.block<match_px_top, acols>(cols / 2 - match_px_top - 1, 0);

        Eigen::Matrix<double, match_px_bot + match_px_top, 1> b;
        b.segment<match_px_bot>(0) = target.segment<match_px_bot>(0);
        b.segment<match_px_top>(match_px_bot) =
            target.segment<match_px_top>(cols / 2 - match_px_top - 1);

        Eigen::Array<double, match_px_bot + match_px_top, 1> w =
            1.0 / (1.0 + AA.col(0).array().sqrt());

        AA.array().colwise() *= w;
        b.array() *= w;

        Eigen::Matrix<double, acols, 1> x =
            (AA.transpose() * AA).inverse() * (AA.transpose() * b);

        mutable_col = A * x;
        // clamp
        mutable_col = (mutable_col.array() * 0.95 + 0.001).matrix();
        mutable_col = mutable_col.unaryExpr([](double x) {
            if (x < 0) return 0.0;
            if (x > 1) return 1.0;
            return x;
        });
    }

    static std::unique_ptr<std::vector<uint8_t>> rgbToByteArray(
        const vecColVec &r, const vecColVec &g, const vecColVec &b) {
        auto arr = std::make_unique<std::vector<uint8_t>>(
            r.size() * cols / 2 * 4, 255);
        auto &ba = *arr;
        for (size_t x = 0; x < r.size(); x++) {
            Eigen::Array<double, cols / 2, 1> R = r[x].array().pow(1 / 2.20);
            Eigen::Array<double, cols / 2, 1> G = g[x].array().pow(1 / 2.20);
            Eigen::Array<double, cols / 2, 1> B = b[x].array().pow(1 / 2.20);
            for (size_t y = 0; y < cols / 2; y++) {
                ba[4 * (x * cols / 2 + y)] = R(y) * 255;
                ba[4 * (x * cols / 2 + y) + 1] = G(y) * 255;
                ba[4 * (x * cols / 2 + y) + 2] = B(y) * 255;
            }
        }
        return arr;
    }

   public:
    PostProcessor()
        : blue_start(ColVec::Zero()),
          red_start(ColVec::Zero()),
          green_start(ColVec::Zero()),
          blue_corrector(ColVec::Zero()),
          red_corrector(ColVec::Zero()),
          green_r_corrector(ColVec::Zero()),
          green_b_corrector(ColVec::Zero()),
          start_n(0) {
        for (size_t i = black_point_calib_start; i < black_point_calib_end;
             i++) {
            size_t j = i - black_point_calib_start;
            blue_corrector(i) = black_point_calib[j * 4];
            green_b_corrector(i) = black_point_calib[j * 4 + 1];
            green_r_corrector(i) = black_point_calib[j * 4 + 2];
            red_corrector(i) = black_point_calib[j * 4 + 3];
        }
    }

    std::unique_ptr<std::vector<uint8_t>> process(
        const std::vector<uint16_t> &raw) {
        vecColVec reds;
        vecColVec greens;
        vecColVec blues;
        reds.reserve(rows / 2);
        greens.reserve(rows / 2);
        blues.reserve(rows / 2);
        for (size_t row = 0; row < rows / 2; row++) {
            ColVec blue =
                ((Eigen::Map<const Eigen::Matrix<uint16_t, cols / 2, 1>,
                             Eigen::Unaligned, Eigen::Stride<1, 2>>(
                      raw.data() + cols * row * 2 + 1)
                      .cast<double>() -
                  blue_corrector) /
                 65536);
            ColVec green_b =
                ((Eigen::Map<const Eigen::Matrix<uint16_t, cols / 2, 1>,
                             Eigen::Unaligned, Eigen::Stride<1, 2>>(
                      raw.data() + cols * row * 2)
                      .cast<double>() -
                  green_b_corrector) /
                 65536);
            ColVec red =
                ((Eigen::Map<const Eigen::Matrix<uint16_t, cols / 2, 1>,
                             Eigen::Unaligned, Eigen::Stride<1, 2>>(
                      raw.data() + cols * (row * 2 + 1))
                      .cast<double>() -
                  red_corrector) /
                 65536);
            ColVec green_r =
                ((Eigen::Map<const Eigen::Matrix<uint16_t, cols / 2, 1>,
                             Eigen::Unaligned, Eigen::Stride<1, 2>>(
                      raw.data() + cols * (row * 2 + 1) + 1)
                      .cast<double>() -
                  green_r_corrector) /
                 65536);

            blue = blue.unaryExpr([](double x) {
                if (x < 0) return 0.0;
                if (x > 1) return 1.0;
                return x;
            });
            red = red.unaryExpr([](double x) {
                if (x < 0) return 0.0;
                if (x > 1) return 1.0;
                return x;
            });
            green_b = green_b.unaryExpr([](double x) {
                if (x < 0) return 0.0;
                if (x > 1) return 1.0;
                return x;
            });
            green_r = green_r.unaryExpr([](double x) {
                if (x < 0) return 0.0;
                if (x > 1) return 1.0;
                return x;
            });

            if (start_n < start_cols) {
                blue_start = (start_n * blue_start + blue) / (start_n + 1);
                red_start = (start_n * red_start + red) / (start_n + 1);
                green_start =
                    (start_n * green_start + (green_b + green_r) * 0.5) /
                    (start_n + 1);
                start_n++;
            } else {
                matchColumnExposure(blue, blue_start, match_thresh);
                matchColumnExposure(red, red_start, match_thresh);
                matchColumnExposure(green_b, green_start, match_thresh);
                matchColumnExposure(green_r, green_start, match_thresh);
                reds.push_back(red);
                greens.push_back((green_b + green_r) / 2);
                blues.push_back(blue);
            }
        }
        return rgbToByteArray(reds, greens, blues);
    }
};

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cerr << "USAGE: demosaic file_1.bin file_2.bin ... file_n.bin"
                  << std::endl;
    }

    PostProcessor pp;
    std::vector<uint16_t> buf(rows * cols);
    const int n = argc - 1;
    for (int f = 0; f < n; f++) {
        std::string filename = argv[f + 1];
        std::ifstream fin(filename, std::ios::in | std::ios::binary);
        fin.read(reinterpret_cast<char *>(buf.data()), rows * cols * 2);
        fin.close();

        filename[filename.size() - 1] = 'g';
        filename[filename.size() - 2] = 'n';
        filename[filename.size() - 3] = 'p';
        auto bytes = pp.process(buf);
        tinypng::PNG png(cols / 2, bytes->size() / (cols / 2) / 4,
                         bytes->data());
        png.writeToFile(filename);
        std::cerr << filename << std::endl;
    }
    return 0;
}
