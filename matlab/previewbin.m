close all;
a = [];
for ind=0:81
    f = fopen(sprintf('../build/im%04d.bin', ind));
    a = [a; fread(f, 16777216)];
    fclose(f);
end
aa = reshape(a(2:2:end), 4096, []) * 256 + reshape(a(1:2:end), 4096, []);
b = aa(2:2:end,1:2:end);
g = (aa(2:2:end,2:2:end) + aa(1:2:end,1:2:end)) / 2;
r = aa(1:2:end,2:2:end);
%imshow(aa' / max(aa(:)));
rgb = cat(3, r, g, b);
imshow(rgb / max(rgb(:)));
