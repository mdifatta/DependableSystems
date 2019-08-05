import os
import sys
import re
import getopt

PROJECT_ROOT = '/home/llfi/Desktop/'
LLFI_ROOT = '/home/llfi/Desktop/llfi/bin/'
IMG_ROOT = '/home/llfi/Desktop/images/'

options, remainder = getopt.getopt(sys.argv[1:], 'cp')

comp_n_instr = False
prof_n_inject = False
for opt, arg in options:
    if opt == '-c':
        comp_n_instr = True
    elif opt == '-p':
        prof_n_inject = True

print('CAVEAT: run this script in home/llfi/Desktop/')
os.system('pwd')

fi_folders = []

for f in os.listdir(PROJECT_ROOT):
    if f.startswith('fi_'):
        if os.path.isdir(os.path.join(PROJECT_ROOT, f)):
            fi_folders.append(f)

print('#>Injection folders found:' + str(fi_folders))

images = []

for i in os.listdir(IMG_ROOT):
    if not os.path.isdir(os.path.join(IMG_ROOT, i)):
        if i.endswith('.pgm'):
            images.append(i)

print('#>Input images found:' + str(images))

cmd_compile = 'clang++ -std=c++03 -O3 -emit-llvm -S feature_extract.cpp'
cmd_instrument = LLFI_ROOT+'instrument --readable '
cmd_profile = './../llfi/bin/profile '
cmd_inject = './../llfi/bin/injectfault '

#img = 'img1.pgm'

for f in fi_folders:
    for img in images:
        if comp_n_instr == True:
            # compile *.cpp file and generate .ll
            print('#>Compiling ' + f + '...')
            os.chdir(PROJECT_ROOT + f)
            os.system('pwd')
            ret = os.system(cmd_compile)
            if ret != 0:
                sys.exit('#>Compile in ' + f + ' error.')
            os.chdir('../')

            print('#>Compile in ' + f + ' successful.')
            os.system('pwd')

            # instrument
            print('#>Instrumenting ' + f +'...') 
            ret = os.system(cmd_instrument + PROJECT_ROOT + f + '/feature_extract.ll')
            if ret != 0:
                sys.exit('#>Instrument in ' + f + ' error.')

            print('#>Instrument in ' + f + ' successful.')

        if prof_n_inject == True:    
            # profile
            print('#>Profiling ' + f + '...')
            os.chdir(PROJECT_ROOT + f)
            os.system('pwd')
            ret = os.system('mv ../images/'+ img +' .')
            if ret != 0:
                sys.exit('#>Move in error.')

            ret = os.system(cmd_profile + './llfi/feature_extract-profiling.exe ' + img)
            if ret != 0:
                os.system('mv ./'+ img +' ../images/')
                sys.exit('#>Profiling in ' + f + ' error.')

            ret = os.system('mv ./'+ img +' ../images/')
            if ret != 0:
                sys.exit('#>Move out error.')

            os.chdir('../')

            print('#>Profile in ' + f + ' successful.')

            # inject
            print('#>Injecting ' + f + '...')
            os.chdir(PROJECT_ROOT + f)
            os.system('pwd')
            ret = os.system('mv ../images/'+ img +' .')
            if ret != 0:
                sys.exit('#>Move in error.')

            ret = os.system(cmd_inject + './llfi/feature_extract-faultinjection.exe ' + img)
            if ret != 0:
                os.system('mv ./'+ img +' ../images/')
                sys.exit('#>Injecting in ' + f + ' error.')

            ret = os.system('mv ./'+ img +' ../images/')
            if ret != 0:
                sys.exit('#>Move out error.')

            os.chdir('../')

            print('#>Injection in ' + f + ' successful.')
