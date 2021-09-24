close all;
a = [];
for ind=10:21
    ind
    f = fopen(sprintf('/home/dllu/pictures/linescan/nyc4/im%04d.bin', ind));
    a = [a; fread(f, 16777216)];
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
imwrite(sqrt(rgb), 'nyc4.jpg');
