clear; clc;

m = 16;
n = 32;
p = 1;

A = zeros(m,n);
B = zeros(n,p);

for i = 1:m
    for j = 1:n
        A(i,j) = i + j - 2;
    end
end

for i = 1:n
    for j = 1:p
        B(i,j) = i + j - 2;
    end
end

C = A*B;
disp(C)
