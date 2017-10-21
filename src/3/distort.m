function letter = distort(letter, num_pixels)
    indices = randsample(numel(letter), num_pixels);
    letter(indices) = -1 * letter(indices);
end