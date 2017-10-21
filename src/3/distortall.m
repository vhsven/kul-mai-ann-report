function letters = distortall(letters, num_pixels)
    for i=1:size(letters, 2)
       letters(:,i) = distort(letters(:,i), num_pixels);
    end
end