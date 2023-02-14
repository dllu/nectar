close all;
a = [];
p = '/home/dllu/pictures/linescan/2023-02-14-12:05:51'
for ind=0:20
    ind
    f = fopen(sprintf([p '/im%06d.bin'], ind));
    ff = fread(f, 16777216);
    a = [a; ff];
    fclose(f);
end
aa = reshape(a(2:2:end), 4096, []) * 256 + reshape(a(1:2:end), 4096, []);
b = aa(2:2:end,1:2:end);
g = (aa(2:2:end,2:2:end) + aa(1:2:end,1:2:end)) / 2;
r = aa(1:2:end,2:2:end);
%imshow(aa' / max(aa(:)));
rgb = cat(3, r, g, b);
rgb = rgb / max(rgb(:));
%imshow(rgb / max(rgb(:)));
rgb = rgb(2048:-1:1, :, :);
imwrite(sqrt(rgb), 'asdf.jpg');
