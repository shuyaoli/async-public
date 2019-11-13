function out = ls(w, x, y)

    out = sum((x * w - y).^2);
end