close all;
clear all;
s = '../calib/im%04d';
a = 0.055;
gt_srgb = dlmread('../calib/srgb.txt') / 255;
gt_xyz = gt_srgb / 12.92;
gt_xyz(gt_srgb > 0.04045) = ((gt_srgb(gt_srgb > 0.04045) + a) ./ (1 + a)).^2.4;

inds = 0:50;
framesize = 16777216;
a = zeros(framesize * length(inds), 1);
for indind = 1:length(inds);
    ind = inds(indind);
    f = fopen(sprintf([s, '.bin'], ind));
    a(framesize * (indind - 1) + 1 : framesize * indind) = fread(f, framesize);
    fclose(f);
end
aa = double(reshape(a(2:2:end), 4096, []) * 256 + reshape(a(1:2:end), 4096, []));
b = aa(2:2:end,1:2:end) / 65536;
gb = aa(2:2:end,2:2:end) / 65536;
gr = aa(1:2:end,1:2:end) / 65536;
r = aa(1:2:end,2:2:end) / 65536;

figure; hold on;
plot(b(1024,:), 'b');
plot(r(1024,:), 'r');
plot(gb(1024,:), 'g');
plot(gr(1024,:), 'k');

boundaries = diff(r(1024,:)).^2 + diff(gr(1024,:)).^2 + diff(gb(1024,:)).^2 + diff(b(1024,:)).^2;
idx = find(boundaries > 3e-3);
start = idx(1);
duration = 200;
pad = 30;

colours = 100;
segmentlength = duration - 2 * pad + 1;
img = zeros(1024 + 4096 + 4096,segmentlength * colours, 3);
calibrations = [];

for colour = 1:colours
    img_segment = (colour - 1) * segmentlength + 1 : colour * segmentlength;
    img(1:1024, img_segment, 1) = gt_xyz(colour, 1);
    img(1:1024, img_segment, 2) = gt_xyz(colour, 2);
    img(1:1024, img_segment, 3) = gt_xyz(colour, 3);
end

for px = 1:2048
    necta_xyyz = zeros(colours, 4);

    for colour = 1:colours
        segment = duration * (colour - 1) + start + pad : duration * colour + start - pad;
        img_segment = (colour - 1) * segmentlength + 1 : colour * segmentlength;
        if segment(end) > size(b,2)
            break
        end
        necta_b = mean(b(px, segment));
        necta_gb = mean(gb(px, segment));
        necta_gr = mean(gr(px, segment));
        necta_r = mean(r(px, segment));

        img(5120 + 2 * px - 1, img_segment, 1) = r(px, segment);
        img(5120 + 2 * px - 0, img_segment, 1) = r(px, segment);
        img(5120 + 2 * px - 1, img_segment, 2) = gb(px, segment);
        img(5120 + 2 * px - 0, img_segment, 2) = gr(px, segment);
        img(5120 + 2 * px - 1, img_segment, 3) = b(px, segment);
        img(5120 + 2 * px - 0, img_segment, 3) = b(px, segment);

        necta_xyyz(colour, 1) = necta_r;
        necta_xyyz(colour, 2) = necta_gr;
        necta_xyyz(colour, 3) = necta_gb;
        necta_xyyz(colour, 4) = necta_b;
    end
    valid = 1:(colour - 1);

    Cr = necta_xyyz(valid,[1 2 4]) \ gt_xyz(valid,:);
    Cb = necta_xyyz(valid,[1 3 4]) \ gt_xyz(valid,:);
    calibrations = [calibrations; [Cr Cb]];
    calibrated_necta_xyz_r = necta_xyyz(valid,[1 2 4]) * Cr;
    calibrated_necta_xyz_b = necta_xyyz(valid,[1 3 4]) * Cb;
    raw_xyz_r = [r(px,:)', gr(px,:)', b(px,:)'];
    raw_xyz_b = [r(px,:)', gb(px,:)', b(px,:)'];
    calibrated_xyz_r = raw_xyz_r * Cr;
    calibrated_xyz_b = raw_xyz_b * Cb;

    if px == 512
        figure;
        subplot(2,2,1); plot(gt_xyz(valid,1), necta_xyyz(valid,1), 'r+', 'MarkerSize', 3, 'LineWidth', 3)
        title('Red channel, uncalibrated');
        xlabel('True red value (sRGB linear)');
        ylabel('Sensor red value (raw values)');
        subplot(2,2,2); plot(gt_xyz(valid,3), necta_xyyz(valid,4), 'b+', 'MarkerSize', 3, 'LineWidth', 3)
        title('Blue channel, uncalibrated');
        xlabel('True blue value (sRGB linear)');
        ylabel('Sensor blue value (raw values)');
        subplot(2,2,3); plot(gt_xyz(valid,2), necta_xyyz(valid,2), 'g+', 'MarkerSize', 3, 'LineWidth', 3)
        title('Green channel (red line), uncalibrated');
        xlabel('True green value (sRGB linear)');
        ylabel('Sensor green value (raw values)');
        subplot(2,2,4); plot(gt_xyz(valid,2), necta_xyyz(valid,3), 'g+', 'MarkerSize', 3, 'LineWidth', 3)
        title('Green channel (blue line), uncalibrated');
        xlabel('True green value (sRGB linear)');
        ylabel('Sensor green value (raw values)');

        figure;
        subplot(1,3,1); plot(gt_xyz(valid,1), calibrated_necta_xyz_r(valid,1), 'r+', 'MarkerSize', 3, 'LineWidth', 3)
        title('Red channel, calibrated_r');
        xlabel('True red value (sRGB linear)');
        ylabel('Sensor red value (calibrated values)');
        subplot(1,3,2); plot(gt_xyz(valid,2), calibrated_necta_xyz_r(valid,2), 'g+', 'MarkerSize', 3, 'LineWidth', 3)
        title('Green channel (red line), calibrated_r');
        xlabel('True green value (sRGB linear)');
        ylabel('Sensor green value (calibrated values)');
        subplot(1,3,3); plot(gt_xyz(valid,3), calibrated_necta_xyz_r(valid,3), 'b+', 'MarkerSize', 3, 'LineWidth', 3)
        title('Blue channel, calibrated_r');
        xlabel('True blue value (sRGB linear)');
        ylabel('Sensor blue value (calibrated values)');

        figure;
        subplot(1,3,1); plot(gt_xyz(valid,1), calibrated_necta_xyz_b(valid,1), 'r+', 'MarkerSize', 3, 'LineWidth', 3)
        title('Red channel, calibrated_b');
        xlabel('True red value (sRGB linear)');
        ylabel('Sensor red value (calibrated values)');
        subplot(1,3,2); plot(gt_xyz(valid,2), calibrated_necta_xyz_b(valid,2), 'g+', 'MarkerSize', 3, 'LineWidth', 3)
        title('Green channel (red line), calibrated_b');
        xlabel('True green value (sRGB linear)');
        ylabel('Sensor green value (calibrated values)');
        subplot(1,3,3); plot(gt_xyz(valid,3), calibrated_necta_xyz_b(valid,3), 'b+', 'MarkerSize', 3, 'LineWidth', 3)
        title('Blue channel, calibrated_b');
        xlabel('True blue value (sRGB linear)');
        ylabel('Sensor blue value (calibrated values)');
    end
    for colour = 1:colours
        segment = duration * (colour - 1) + start + pad : duration * colour + start - pad;
        img_segment = (colour - 1) * segmentlength + 1 : colour * segmentlength;
        img(1024 + 2 * px - 1, img_segment, 1) = calibrated_xyz_r(segment, 1);
        img(1024 + 2 * px - 0, img_segment, 1) = calibrated_xyz_b(segment, 1);
        img(1024 + 2 * px - 1, img_segment, 2) = calibrated_xyz_r(segment, 2);
        img(1024 + 2 * px - 0, img_segment, 2) = calibrated_xyz_b(segment, 2);
        img(1024 + 2 * px - 1, img_segment, 3) = calibrated_xyz_r(segment, 3);
        img(1024 + 2 * px - 0, img_segment, 3) = calibrated_xyz_b(segment, 3);
    end
end
img(img < 0) = 0;
img(img > 1) = 1;
dlmwrite('calibration.txt', calibrations);
imwrite(img, 'calibration.png');
%g = (gb + gr) / 2;
