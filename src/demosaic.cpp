#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "Condat_TV_1D_v2.c"
#include "tinypng.h"

constexpr size_t rows = 2048;
constexpr size_t cols = 4096;
constexpr size_t start_cols = 100;
constexpr size_t match_px_top = 350;
constexpr size_t match_px_bot = 240;

using ColVec = Eigen::Matrix<float, cols / 2, 1>;
using vecColVec = std::vector<ColVec, Eigen::aligned_allocator<ColVec>>;

constexpr size_t black_point_calib_start = 1294;
constexpr size_t black_point_calib_end = 1314;
constexpr size_t calib_n = black_point_calib_end - black_point_calib_start;
constexpr float black_point_calib[] = {
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
    Eigen::Array<float, cols / 2, -1> reds;
    Eigen::Array<float, cols / 2, -1> greens_r;
    Eigen::Array<float, cols / 2, -1> greens_b;
    Eigen::Array<float, cols / 2, -1> blues;

    ColVec red_corrector;
    ColVec green_r_corrector;
    ColVec green_b_corrector;
    ColVec blue_corrector;
    size_t col_ind;
    size_t frame_ind;

    /**
     * Each column may have a slightly different exposure due to electrical
     * noise in the sensor readout. Correct for this error by performing a
     * linear regression against the *_start columns, which are averaged over
     * start_cols. The regression variables are intensity and y value (index
     * along the col).
     */
    static void matchColumnExposure(Eigen::Ref<ColVec> mutable_col,
                                    const ColVec &target) {
        constexpr size_t acols = 3;
        Eigen::Matrix<float, cols / 2, acols> A;
        A.col(0) = mutable_col;
        A.col(1).setLinSpaced(0, 1);
        A.col(2).setOnes();

        Eigen::Matrix<float, match_px_bot + match_px_top, acols> AA;
        AA.block<match_px_bot, acols>(0, 0) =
            A.block<match_px_bot, acols>(0, 0);
        AA.block<match_px_top, acols>(match_px_bot, 0) =
            A.block<match_px_top, acols>(cols / 2 - match_px_top - 1, 0);

        Eigen::Matrix<float, match_px_bot + match_px_top, 1> b;
        b.segment<match_px_bot>(0) = target.segment<match_px_bot>(0);
        b.segment<match_px_top>(match_px_bot) =
            target.segment<match_px_top>(cols / 2 - match_px_top - 1);

        Eigen::Array<float, match_px_bot + match_px_top, 1> w =
            1.0 / (1.0 + AA.col(0).array().sqrt());

        AA.array().colwise() *= w;
        b.array() *= w;

        Eigen::Matrix<float, acols, 1> x =
            (AA.transpose() * AA).inverse() * (AA.transpose() * b);

        mutable_col = A * x;
    }

    static std::unique_ptr<std::vector<uint8_t>> rgbToByteArray(
        Eigen::Ref<Eigen::Array<float, cols / 2, -1>> r,
        Eigen::Ref<Eigen::Array<float, cols / 2, -1>> g,
        Eigen::Ref<Eigen::Array<float, cols / 2, -1>> b) {
        auto arr = std::make_unique<std::vector<uint8_t>>(rows * cols, 255);
        auto &ba = *arr;
        auto clam = [](float y) {
            if (y < 0) return (float)0;
            if (y > 1) return (float)1;
            return y;
        };
        Eigen::Array<float, cols / 2, -1> R =
            r.unaryExpr(clam).array().pow(1 / 2.20);
        Eigen::Array<float, cols / 2, -1> G =
            g.unaryExpr(clam).array().pow(1 / 2.20);
        Eigen::Array<float, cols / 2, -1> B =
            b.unaryExpr(clam).array().pow(1 / 2.20);
        for (size_t x = 0; x < rows / 2; x++) {
            for (size_t y = 0; y < cols / 2; y++) {
                ba[4 * (x * cols / 2 + y)] = R(y, x) * 255;
                ba[4 * (x * cols / 2 + y) + 1] = G(y, x) * 255;
                ba[4 * (x * cols / 2 + y) + 2] = B(y, x) * 255;
            }
        }
        return arr;
    }

   public:
    PostProcessor(size_t n)
        : reds(cols / 2, rows / 2 * n),
          greens_r(cols / 2, rows / 2 * n),
          greens_b(cols / 2, rows / 2 * n),
          blues(cols / 2, rows / 2 * n),
          red_corrector(ColVec::Zero()),
          green_r_corrector(ColVec::Zero()),
          green_b_corrector(ColVec::Zero()),
          blue_corrector(ColVec::Zero()),
          col_ind(0),
          frame_ind(0) {
        for (size_t i = black_point_calib_start; i < black_point_calib_end;
             i++) {
            size_t j = i - black_point_calib_start;
            blue_corrector(i) = black_point_calib[j * 4];
            green_b_corrector(i) = black_point_calib[j * 4 + 1];
            green_r_corrector(i) = black_point_calib[j * 4 + 2];
            red_corrector(i) = black_point_calib[j * 4 + 3];
        }
    }

    std::unique_ptr<std::vector<uint8_t>> pop() {
        Eigen::Array<float, cols / 2, -1> greens =
            0.5 * (greens_r.block<cols / 2, rows / 2>(0, frame_ind * rows / 2) +
                   greens_b.block<cols / 2, rows / 2>(0, frame_ind * rows / 2));
        auto bytes = rgbToByteArray(
            reds.block<cols / 2, rows / 2>(0, frame_ind * rows / 2), greens,
            blues.block<cols / 2, rows / 2>(0, frame_ind * rows / 2));
        frame_ind++;
        return bytes;
    }

    void push(const std::vector<uint16_t> &raw) {
        for (size_t row = 0; row < rows / 2; row++) {
            blues.col(col_ind) =
                ((Eigen::Map<const Eigen::Matrix<uint16_t, cols / 2, 1>,
                             Eigen::Unaligned, Eigen::Stride<1, 2>>(
                      raw.data() + cols * row * 2 + 1)
                      .cast<float>() -
                  blue_corrector) /
                 65536);
            greens_b.col(col_ind) =
                ((Eigen::Map<const Eigen::Matrix<uint16_t, cols / 2, 1>,
                             Eigen::Unaligned, Eigen::Stride<1, 2>>(
                      raw.data() + cols * row * 2)
                      .cast<float>() -
                  green_b_corrector) /
                 65536);
            reds.col(col_ind) =
                ((Eigen::Map<const Eigen::Matrix<uint16_t, cols / 2, 1>,
                             Eigen::Unaligned, Eigen::Stride<1, 2>>(
                      raw.data() + cols * (row * 2 + 1))
                      .cast<float>() -
                  red_corrector) /
                 65536);
            greens_r.col(col_ind) =
                ((Eigen::Map<const Eigen::Matrix<uint16_t, cols / 2, 1>,
                             Eigen::Unaligned, Eigen::Stride<1, 2>>(
                      raw.data() + cols * (row * 2 + 1) + 1)
                      .cast<float>() -
                  green_r_corrector) /
                 65536);
            col_ind++;
        }
    }
    void process() {
        ColVec red_start = reds.leftCols(1000).rowwise().mean();
        ColVec green_start = (greens_r.leftCols(1000).rowwise().mean() +
                              greens_b.leftCols(1000).rowwise().mean()) *
                             0.5;
        ColVec blue_start = blues.leftCols(1000).rowwise().mean();
        const size_t n = reds.cols();
        Eigen::ArrayXf tmp0(n);
        Eigen::ArrayXf tmp1(n);

        for (size_t c = 0; c < n; c++) {
            matchColumnExposure(reds.col(c), red_start);
            matchColumnExposure(greens_r.col(c), green_start);
            matchColumnExposure(greens_b.col(c), green_start);
            matchColumnExposure(blues.col(c), blue_start);

            TV1D_denoise_v2(
                reds.block<calib_n + 2, 1>(black_point_calib_start, c).data(),
                tmp0.data(), calib_n + 2, 0.02);
            reds.block<calib_n + 2, 1>(black_point_calib_start, c) = tmp0;

            TV1D_denoise_v2(
                greens_r.block<calib_n + 2, 1>(black_point_calib_start, c)
                    .data(),
                tmp0.data(), calib_n + 2, 0.02);
            greens_r.block<calib_n + 2, 1>(black_point_calib_start, c) = tmp0;

            TV1D_denoise_v2(
                greens_b.block<calib_n + 2, 1>(black_point_calib_start, c)
                    .data(),
                tmp0.data(), calib_n + 2, 0.02);
            greens_b.block<calib_n + 2, 1>(black_point_calib_start, c) = tmp0;

            TV1D_denoise_v2(
                blues.block<calib_n + 2, 1>(black_point_calib_start, c).data(),
                tmp0.data(), calib_n + 2, 0.02);
            blues.block<calib_n + 2, 1>(black_point_calib_start, c) = tmp0;
        }
        for (size_t r = 0; r < cols / 2; r++) {
            tmp1 = reds.row(r);
            TV1D_denoise_v2(tmp1.data(), tmp0.data(), n, 0.005);
            reds.row(r) = tmp0;

            tmp1 = greens_r.row(r);
            TV1D_denoise_v2(tmp1.data(), tmp0.data(), n, 0.005);
            greens_r.row(r) = tmp0;

            tmp1 = greens_b.row(r);
            TV1D_denoise_v2(tmp1.data(), tmp0.data(), n, 0.005);
            greens_b.row(r) = tmp0;

            tmp1 = blues.row(r);
            TV1D_denoise_v2(tmp1.data(), tmp0.data(), n, 0.01);
            blues.row(r) = tmp0;
        }
    }
};

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cerr << "USAGE: demosaic file_1.bin file_2.bin ... file_n.bin"
                  << std::endl;
    }

    std::vector<uint16_t> buf(rows * cols);
    const int n = argc - 1;

    PostProcessor pp(n);
    for (int f = 0; f < n; f++) {
        std::string filename = argv[f + 1];
        std::ifstream fin(filename, std::ios::in | std::ios::binary);
        fin.read(reinterpret_cast<char *>(buf.data()), rows * cols * 2);
        fin.close();
        pp.push(buf);
    }
    pp.process();
    for (int f = 0; f < n; f++) {
        std::string filename = argv[f + 1];
        filename[filename.size() - 1] = 'g';
        filename[filename.size() - 2] = 'n';
        filename[filename.size() - 3] = 'p';
        auto bytes = pp.pop();
        tinypng::PNG png(cols / 2, bytes->size() / (cols / 2) / 4,
                         bytes->data());
        png.writeToFile(filename);
        std::cerr << filename << std::endl;
    }
    return 0;
}
