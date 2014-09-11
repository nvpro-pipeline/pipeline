
import sys;
import re;


fin = open(sys.argv[1])
lines = fin.readlines()
fin.close()

framesPassed = 0
totalT = 0.0

for iline in range(len(lines)):
   cfgline = lines[iline]
   
   important_part = re.sub( '^([^#]*).*$', "\\1", cfgline )
   important_part = re.sub('\s*$', "", important_part)
   
   if re.match('[^\s]', important_part):
     if re.match('^[0-9]*,[\-]?[0-9\.]*e?[\-]?[0-9]*', important_part):
       frameT = re.sub( '^[^,]*,([\-]?[0-9\.]*e?[\-]?[0-9]*)', "\\1", important_part )
       if framesPassed:
         totalT = totalT + float(frameT)
       framesPassed = framesPassed + 1
	 
valFPS = framesPassed / totalT
valFPSint = int(valFPS)

sys.exit( valFPSint )