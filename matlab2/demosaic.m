clear all;
close all;
superinds = 20:101
speed = []
for superind = superinds

inds = [superind]
raw = zeros(4096, 2048 * size(inds, 2));
for ind=inds
    f = fopen(sprintf('/home/dllu/pictures/linescan/nyc4/im%04d.bin', ind));
    a = fread(f, 16777216);
    aa = reshape(a(2:2:end), 4096, []) * 256 + reshape(a(1:2:end), 4096, []);

    indx = 2048 * (ind - inds(1));
    raw(:, (indx+1):(indx+2048)) = aa;
    fclose(f);
end

r = raw(1:2:end, 2:2:end);
b = raw(2:2:end, 1:2:end);
g1 = raw(2:2:end,2:2:end);
g2 = raw(1:2:end,1:2:end);
rgb = cat(3, r, g1, b);
rgb = rgb / max(rgb(:));
imshow(sqrt(rgb))
%axis equal;

convomg = zeros(1, 21);

for yy = 1220:3:1250
    gg1 = g1(1224,:);
    gg2 = g2(1224,:);

    %{
    gg1 = gg1(1900:2050);
    gg2 = gg2(1900:2050);
    %}

    %{
    figure;
    plot(gg1 - mean(gg1)); hold all;
    plot(gg2 - mean(gg2))
    axis tight;
    %}

    gconv = fftconv(gg1, flip(gg2));
    %gconv2 = fftconv(gg1, flip(gg1));
    n = size(gg1, 2);
    gconv = gconv((n-10):(n+10));
    gconv = gconv - mean(gconv);
    convomg += gconv;
    %gconv2 = gconv2((n-10):(n+10));
end
%gconv2 = gconv2 - mean(gconv2);
shift = meanshift(convomg, 2, 11);
speed = [speed, (shift - 11)];
%{
figure;
plot((1:21) - (shift - 11), gconv, 'b')
hold all;
plot(1:21, gconv2, 'r')
plot(11, gconv2(11), 'or')
axis tight;
%}

end

figure;
plot(superinds, 1 ./ speed)
axis tight
xlabel('frame');
ylabel('train speed');
