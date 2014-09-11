
import sys;
import re;
import os;
import shutil;

configFileName = "__config.cfg"
templateNameList = []
templateDescList = []
curTemplateName = ""

variableList = []
variableDescList = []
variableContents = []

strList = []
strDescList = []
strContents = []


testFolderName = ""
oldFileList = []
newFileList = []

curModuleDirs = []
curModuleNames = []
curSelectedModuleDir = ""

newModuleDirs = []
curSelectedNewModuleDir = ""
newModuleName = ""


iline = 0
icurparse = 0

### Check that a variable name is valid ###
def isValidVariableName(varname):
  if not re.search('[^A-Za-z0-9_]', varname) and not re.match('^[0-9]', varname):
    return 1
  else:
    return 0
  
### Get input for a variable declared in the config file ###
def promptVariables():
  for index in range(len(variableList)):
    print("Please enter ", variableDescList[index], ":")
    while 1:
      curVar = input("")
      if isValidVariableName(curVar):
        variableContents.append( re.sub('/s*$', "", curVar) )
        break
      else:
        print(curVar, " is not a valid variable name, please enter it again")
  return

### Get input for a string declared in the config file ###
def promptStrings():
  for index in range(len(strList)):
    print("Please enter ", strDescList[index], ":")
    curStr = input("")
    strContents.append( re.sub('/n$', "", curStr) )
  return

### Insert parsed variables into a string ###
def insertVars(replacee):
  for index in range(len(variableList)):
    replacee = re.sub( '##%s##' % variableList[index], variableContents[index], replacee)
  return replacee

### Insert parsed string variables into a string ###
def insertStrs(replacee):
  for index in range(len(strList)):
    replacee = re.sub( '##%s##' % strList[index], strContents[index], replacee)
  return replacee

### Get all the template names and their descriptions ###
def gatherTemplateConfigs():
  for subItem in os.listdir():
    #If the item is a directory then this is potentially a template
    if os.path.isdir(subItem):
      for subItemSubDir in os.listdir(subItem):
        #If a config file is in the subdirectory then we have a template
        if os.path.isfile(subItem + "/" + subItemSubDir) and subItemSubDir == configFileName:
          templateNameList.append(subItem)
          #Read the file to find the description
          cfgin = open(subItem + "/" + subItemSubDir)
          lines = cfgin.readlines()
          foundDesc = 0
          for iline in range(len(lines)):
            if re.match( '\[description\]', lines[iline] ):
              foundDesc = 1
              #Make sure the [description] heading is not the last line in the file
              assert( iline+1 < len(lines)-1 )
              templateDescList.append( re.sub('\s*$', "", lines[iline+1] ) )
          #A description must exist in the file
          assert( foundDesc )
          cfgin.close()
  return
  
gatherTemplateConfigs()

### Prompt the user the desired test template ###
print("Please select the number of the test template you would like to use:")
for index in range(len(templateNameList)):
  print(index+1, ") ", templateDescList[index])

while 1:
  inputLine = input("")
  if re.match('[^0-9]', inputLine):
    print("Invalid input, please try again")
    continue
  templateIndex = int(inputLine)
  if templateIndex > 0 and templateIndex <= len(templateNameList):
    break
  else:
    print("No template of that number exists, please try again")
curTemplateName = templateNameList[templateIndex-1]

fin = open(curTemplateName + "/" + configFileName)
lines = fin.readlines()
fin.close()

for iline in range(len(lines)):
   cfgline = lines[iline]
   
   if re.match( '^\[.*\]$', cfgline ):
     if re.match( '\[variables\]', cfgline ):
        icurparse = 1
     elif re.match( '\[filenames\]', cfgline ):
        promptVariables()
        promptStrings()
        icurparse = 2
     elif re.match( '\[folder\]', cfgline ):
        icurparse = 3
     elif re.match( '\[modules\]', cfgline ):
        icurparse = 4
     continue

### Parse the variables in the config file ###
   if icurparse == 1:
      matchObj = re.match('^\s*[A-Za-z]*', cfgline)
      if matchObj:
         varType = re.sub( '\s*', "", matchObj.group() )
         if varType == "var":
           #extract the remainder of the line following the 'var' type qualifier
           varItself = re.sub('%s\s*' % matchObj.group(), "", cfgline)

           #extract the variable description following the variable name
           variableDescList.append( re.sub( '^[A-Za-z_]*\s*(.*)\n$', "\\1", varItself ) )

           #extract the name of the variable
           variableList.append( re.match('^[A-Za-z_]*', varItself).group() )

         if varType == "str":
           #extract the remainder of the line following the 'str' type qualifier
           varItself = re.sub('%s\s*' % matchObj.group(), "", cfgline)

           #extract the string description following the string name
           strDescList.append( re.sub( '^[A-Za-z_]*\s*(.*)\n$', "\\1", varItself ) )

           #extract the name of the string
           strList.append( re.match('^[A-Za-z_]*', varItself).group() )

### Parse source file renaming rules ###
   elif icurparse == 2:
      #read a filename (a string that doesn't contain \ / : * ? " < > |) from the beginning of the line
      filename = re.match( '^[^\\/\:\*\?\"\<\>\|]*', cfgline ).group()

      #read the string directly following filename and illegal file name characters
      replacee = re.sub( '^[^\\/\:\*\?\"\<\>\|]*[\\/\:\*\?\"\<\>\|\s]*(.*)\s*$', "\\1", cfgline )

      oldFileList.append(filename)
      newFileList.append( insertVars(replacee) )

### Parse the name of the folder that will be used for the generated test ###
   elif icurparse == 3:
      #Remove the end of line character at end of the line
      testFolderName = re.sub( '\s$', "", insertVars(cfgline) )

### Parse the available modules in which to contain the generated test ###
   elif icurparse == 4:
      #remove end line character(s)
      noNewLine = re.sub('\s*$', "", cfgline)
      curModuleDirs.append( noNewLine )
      #get the top folder of the path
      curModuleNames.append( re.sub('^.*/([^/]*)$', "\\1", noNewLine ) )
      #get the path minus the top folder
      newModuleDir = re.sub('(^.*/[^/]*)/[^/]*$', "\\1", noNewLine )
      moduleFound = 0
      for index in range(len(newModuleDirs)):
        if newModuleDirs[index] == newModuleDir:
          moduleFound = 1
          break
      if not moduleFound or len(newModuleDirs) == 0:
        newModuleDirs.append( newModuleDir )

### Prompt the user the desired module to contain the generated test ###
print("Please select which module you would like the test to be contained in, or enter 0 if you wish to create your own module:")
for index in range(len(curModuleNames)):
  print(index+1, ") ", curModuleNames[index])

while 1:
  inputLine = input("")
  if re.match('[^0-9]', inputLine):
    print("Invalid input, please try again")
    continue
  moduleIndex = int(inputLine)
  if moduleIndex >= 0 and moduleIndex <= len(curModuleNames):
    break
  else:
    print("No module of that number exists, please try again")

if moduleIndex != 0:
  curSelectedModuleDir = curModuleDirs[moduleIndex-1]
else:
  #The user has chosen to create a new module
  print("Please select which directory you would like to create a new module in")
  for index in range(len(newModuleDirs)):
    print(index+1, ") ", newModuleDirs[index])
  while 1:
    inputLine = input("")
    if re.match('[^0-9]', inputLine):
      print("Invalid input, please try again")
      continue
    moduleDirIndex = int(inputLine)
    if moduleDirIndex > 0 and moduleDirIndex <= len(newModuleDirs):
      break
    else:
      print("No module directory of that number exists, please try again")
  curSelectedNewModuleDir = newModuleDirs[moduleDirIndex-1]
  print("Please enter the name of the new module")
  while 1:
    newModuleName = input("")
    if not re.match( '[\\/\:\*\?\"\<\>\|]', newModuleName ):
      break
    else:
      print("Invalid folder name, please try again")
  
  #We need a sample CMakeLists.txt file from any existing module in the modules directory
  anySubdir = ""
  for subItem in os.listdir(curSelectedNewModuleDir):
    curSubdir = curSelectedNewModuleDir + "/" + subItem
    if os.path.isdir(curSubdir):
      anySubdir = curSubdir
      break
  #The modules directory must contain subdirectories, and they must contain a CMakeLists.txt
  assert( anySubdir )
  assert( os.path.isfile(anySubdir + "/CMakeLists.txt") )
  
  
  #We must create the subdirectory for the new module, and this will thus be our current module
  curSelectedModuleDir = curSelectedNewModuleDir + "/" + newModuleName
  os.mkdir(curSelectedModuleDir)
  #Copy the module CMakeLists.txt file
  shutil.copyfile(anySubdir + "/CMakeLists.txt", curSelectedModuleDir + "/CMakeLists.txt")

### Create the target directory ###
os.mkdir(curSelectedModuleDir + "/" + testFolderName)
#get a list of all the files in the source directory
filesInDir = os.listdir(curTemplateName)

for curDirFileIndex in range( len(filesInDir) ):
  foundFile = 0
  for curFileIndex in range( len(oldFileList) ):
    #if the source directory contains a file that we've defined a rule for, we apply the rule to the contents of the file
    if filesInDir[curDirFileIndex] == oldFileList[curFileIndex]:
      fin = open(curTemplateName + "/" + oldFileList[curFileIndex], "r")
      filebuf = fin.read()
      fin.close()
      newfilebuf = insertVars(filebuf)
      newfilebuf = insertStrs(newfilebuf)
      fout = open(curSelectedModuleDir + "/" + testFolderName + "/" + newFileList[curFileIndex], "w")
      fout.write(newfilebuf)
      fout.close()
      foundFile = 1
      break
  
    #if the source directory doesn't contain any files we've defined a rule for and it's not the config file, just copy the file to the target directory
  if( not foundFile and filesInDir[curDirFileIndex] != configFileName ):
    shutil.copyfile(curTemplateName + "/" + filesInDir[curDirFileIndex], curSelectedModuleDir + "/" + testFolderName + "/" + filesInDir[curDirFileIndex])
