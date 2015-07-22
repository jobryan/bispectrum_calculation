'''
Fix the number formatting for alpha_beta / flr1r2 output files. E.g.,

    1.234-100 --> 1.234e-100

'''

import re
import sys

def main(inFilename, outFilename, headerLines=3):
    lineCount = 0
    r1 = '(-?\d*\.?\d*)E?(\+|-)(\d*)'
    r2 = r'\1E\2\3'
    with open(inFilename, 'r') as inFile:
        with open(outFilename, 'w') as outFile:
            while lineCount < headerLines:
                outFile.write(inFile.next())
                lineCount += 1
            for line in inFile:
                splitLine = re.split('    ', line)
                splitLine2 = [re.sub(r1,r2,item) for item in splitLine]
                modifiedLine = '    '.join(splitLine)
                outFile.write(re.sub(r1,r2,modifiedLine))
                lineCount += 1
                if (lineCount % 100 == 0):
                    sys.stdout.write('\r'+'read and wrote %i lines' % lineCount)
    print ''
    print 'finished fixing %i lines' % lineCount
    print 'written to %s' % outFilename

if __name__=='__main__':
    if (len(sys.argv) > 1):
        main(sys.argv[1], sys.argv[2])
    else:
        print 'no file given to fix number formatting'