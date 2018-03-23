s = '/mnt/data8/pictures/linescan/17-12-03-23-11-30-utc/im%04d';
close all;
a = [];
png = [];
for ind=[29]
    f = fopen(sprintf([s, '.bin'], ind));
    %ff = imread(sprintf([s, '.png'], ind));
    %png = cat(1, png, ff);
    a = [a; fread(f, 16777216)];
    fclose(f);
end
aa = double(reshape(a(2:2:end), 4096, []) * 256 + reshape(a(1:2:end), 4096, []));
b = aa(2:2:end,1:2:end);
gb = aa(2:2:end,2:2:end);
gr = aa(1:2:end,1:2:end);
g = (gb + gr) / 2;
r = aa(1:2:end,2:2:end);
%{
%imshow(aa' / max(aa(:)));
rgb = cat(3, r', g', b');

%figure(1); imshow(sqrt(rgb / 65536));
%figure(2); imshow(png);

i1 = 1294;
i2 = 1315;
in = i2-i1-1;

corrector = [];
for x= i1+1:i2-1
    %{
    if x == 1311
         figure(3);
         plot(r(x,:), 'r');
         hold on; plot((r(i2,:) * (x-i1) / in + r(i1,:) * (i2-x)/in), 'b');
         figure(4);
         plot(r(x,:), (r(i2,:) * (x-i1) / in + r(i1,:) * (i2-x)/in) - r(x,:), '.');
         axis equal tight;

     end
     %}
     mb = median(b(x,:) - (b(i2,:) * (x-i1) / in + b(i1,:) * (i2-x)/in));
     mgb = median(gb(x,:) - (gb(i2,:) * (x-i1) / in + gb(i1,:) * (i2-x)/in));
     mgr = median(gr(x,:) - (gr(i2,:) * (x-i1) / in + gr(i1,:) * (i2-x)/in));
     mr = median(r(x,:) - (r(i2,:) * (x-i1) / in + r(i1,:) * (i2-x)/in));
     b(x,:) = b(x,:) - mb;
     gb(x,:) = gb(x,:) - mgb;
     gr(x,:) = gr(x,:) - mgr;
     r(x,:) = r(x,:) - mr;
     corrector = [corrector; mb mgb mgr mr];
end

g = (gb + gr) / 2;
rgb = cat(3, r', g', b');

%figure(2); imshow(sqrt(rgb / 65536));
rgb = rgb - min(rgb(:));
rgb = rgb / max(rgb(:));
rgbd = rgb;
parfor c = 1:3
    for y = 1:2048
        [rgbd(:,y,c), ~, ~, ~] = tvdip(rgb(:,y,c), 0.02, 0, 1e-3, 10);
    end
endparfor
rgbd = rgbd - min(rgbd(:));
rgbd = rgbd / max(rgbd(:));
figure(2); imshow(sqrt(rgbd));
imwrite(sqrt([rgbd, rgb]), 'zxcv.png');
