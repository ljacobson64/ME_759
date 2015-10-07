clear; clc;

m = 15;
n = 15;
p = 15;

A = zeros(m,n);
B = zeros(n,p);

for i = 0:m-1
    for j = 0:n-1
        A(i+1,j+1) = 10*i + j;
    end
end

for i = 0:n-1
    for j = 0:p-1
        B(i+1,j+1) = 10*i + j;
    end
end

C = A*B;
disp(C)
