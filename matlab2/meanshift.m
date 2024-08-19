function peak = meanshift(y, sigma, peak_guess)
    peak = peak_guess;
    x = 1:size(y, 2);

    for iter = 1:10
        weights = exp(-(x - peak).^2 / sigma^2) .* y;
        off = sum(weights .* x) / sum(weights);
        peak = off;
    end
endfunction
