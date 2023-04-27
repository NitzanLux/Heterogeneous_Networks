from subprocess import Popen
import sys
from art import tprint

params = ' '.join(sys.argv[1:])
lines = params.split('\n')
# print(lines)
for i in lines:
    process = Popen(i, shell=True)
    stdout, stderr = process.communicate()
    print(stdout,file=sys.stdout)
    print(stderr,file=sys.stderr)
    print(i)
    tprint(f'Finished')
#