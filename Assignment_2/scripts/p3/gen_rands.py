import random
import sys

nrand_exp = int(sys.argv[1])
nrands = 2**nrand_exp
rand_min = -2**31
rand_max = 2**31-1

if nrand_exp < 10:
    nrand_exp_str = "0"+str(nrand_exp)
else:
    nrand_exp_str = str(nrand_exp)


writer = open("zinp_"+nrand_exp_str,'w')

for i in range(nrands):
    writer.write("%s\n" % random.randint(rand_min, rand_max))

writer.close()
