// Copyright (c) 2009-2016, NVIDIA CORPORATION. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/Scene.h>
#include <dp/sg/ui/ViewState.h>
#include <dp/sg/io/IO.h>
#include <dp/sg/algorithm/Optimize.h>
#include <QApplication>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QProcess>
#include <QStringList>

// optimize
#include <dp/sg/algorithm/AnalyzeTraverser.h>
#include <dp/sg/algorithm/AppTraverser.h>
// DAR HACK Fix build: #include <dp/sg/algorithm/ApplyTransformTraverser.h>
#include <dp/sg/algorithm/CombineTraverser.h>
#include <dp/sg/algorithm/DestrippingTraverser.h>
#include <dp/sg/algorithm/EliminateTraverser.h>
#include <dp/sg/algorithm/IdentityToGroupTraverser.h>
#include <dp/sg/algorithm/NormalizeTraverser.h>
#include <dp/sg/algorithm/SearchTraverser.h>
#include <dp/sg/algorithm/StatisticsTraverser.h>
#include <dp/sg/algorithm/TriangulateTraverser.h>
#include <dp/sg/algorithm/UnifyTraverser.h>

#include <cstdlib>

#if defined(DP_OS_LINUX)
inline int  _putenv( char const *value )
{
  return putenv( const_cast<char*>(value) );
}
#endif

using namespace std;

int main(int argc, char *argv[])
{
  QApplication qapp(argc, argv);

  if (argc < 2 )
  {
    std::cerr << "usage: makenbf inputfile [extension]" << std::endl << std::endl;
    std::cerr << "This will write basename(inputfile).extension. If extension is omitted, nbf is written." << std::endl;
    exit(1);
  }

  dp::sg::ui::ViewStateSharedPtr viewStateHandle;

  // make sbf saver include shader sourcecode
  _putenv( "DP_DAE_WORKAROUNDS=ADD_DP_MATERIAL:DP_MATERIAL_ONLY:MATERIAL_TO_VERTEX_COLORS" );

  QString inputName, outputName;

  bool eraseTmp = false;

  QFileInfo fileInfo( argv[1] );

  QString fileExt;
  if( argc > 2 )
  {
    fileExt = argv[2];
  }
  else
  {
    fileExt = "nbf";
  }


  QString baseName = fileInfo.absolutePath().append("/").append( fileInfo.baseName() );
  inputName = baseName + QString(".").append( fileInfo.suffix() );
  outputName = QString(baseName).append(".").append(fileExt);


  std::cout << "loading" << std::endl;
  if ( !(viewStateHandle = dp::sg::io::loadScene( std::string(inputName.toLocal8Bit()) ) ) )
  {
    std::cerr << "Error loading scene\n";
    return 2;
  }

  std::cout << "optimizing..." << std::endl;
  dp::sg::algorithm::optimizeScene( viewStateHandle->getScene(), true, true
                                  , dp::sg::algorithm::CombineTraverser::Target::ALL
                                  , dp::sg::algorithm::EliminateTraverser::Target::ALL
                                  , dp::sg::algorithm::UnifyTraverser::Target::ALL, FLT_EPSILON );

  int return_code = 0;
  std::cout << "saving scene to " << outputName.toLocal8Bit().data() << std::endl;
  if (!dp::sg::io::saveScene( std::string(outputName.toLocal8Bit()), viewStateHandle ))
  {
    std::cerr << "Error saving scene\n";
    return_code = 2;
  }

  std::cout << "deleting..." << std::endl;
  viewStateHandle.reset();

  if (!return_code)
  {
    std::cout << "done" << std::endl;
  }

  return return_code;
}
