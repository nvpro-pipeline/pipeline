// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <iostream>
#include <tinyxml.h>
#include <boost/program_options.hpp>
#include <dp/DP.h>
#include <dp/fx/mdl/inc/MaterialBuilder.h>
#include <dp/util/File.h>
#include "GLSLBuilder.h"
#include "XMLBuilder.h"

int main( int argc, char *argv[] )
{
  boost::program_options::options_description od("Usage: mdl2xml");
  od.add_options()
    ( "file", boost::program_options::value<std::string>(), "single file to handle" )
    ( "help",                                               "show help")
    ( "path", boost::program_options::value<std::string>(), "path to multiple files to handle" )
    ( "root", boost::program_options::value<std::string>(), "root path of the material package" )
    ;

  boost::program_options::variables_map opts;
  boost::program_options::store( boost::program_options::parse_command_line( argc, argv, od ), opts );

  if ( !opts["help"].empty() )
  {
    std::cout << od << std::endl;
    return( 0 );
  }
  if ( opts["file"].empty() && opts["path"].empty() )
  {
    std::cout << argv[0] << " : at least argument --file or arguments --path is needed!" << std::endl;
    return( 0 );
  }

  std::string file, root, path;
  if ( !opts["file"].empty() )
  {
    if ( !opts["path"].empty() )
    {
      std::cout << argv[0] << " : argument --file and argument --path exclude each other!" << std::endl;
      return( 0 );
    }
    file = opts["file"].as<std::string>();
  }
  if ( !opts["root"].empty() )
  {
    root = opts["root"].as<std::string>();
    if ( ! dp::util::directoryExists( root ) )
    {
      std::cout << argv[0] << " : root <" << root << "> not found!" << std::endl;
      return( 0 );
    }
    if ( ( root.back() != '\\' ) && ( root.back() != '/' ) )
    {
      root.push_back( '\\' );
    }
  }
  if ( !opts["path"].empty() )
  {
    path = opts["path"].as<std::string>();
  }

  std::vector<std::string> files;
  if ( !file.empty() )
  {
    files.push_back( root + file );
    if ( ! dp::util::fileExists( files.back() ) )
    {
      std::cout << argv[0] << " : file <" << files.back() << "> not found!" << std::endl;
      return( 0 );
    }
    if ( dp::util::getFileExtension( files.back() ) != ".mdl" )
    {
      std::cout << argv[0] << " : file <" << files.back() << "> is not an mdl file!" << std::endl;
      return( 0 );
    }
  }
  else
  {
    path = root + path;
    if ( ! dp::util::directoryExists( path ) )
    {
      std::cout << argv[0] << " : path <" << path << "> not found!" << std::endl;
      return( 0 );
    }
    dp::util::findFilesRecursive( ".mdl", path, files );
  }

  if ( files.empty() )
  {
    std::cerr << argv[0] << " : No files found!";
    return( -1 );
  }

  dp::util::FileFinder fileFinder;
  if ( !root.empty() )
  {
    DP_ASSERT( ( root.back() == '\\' ) || ( root.back() == '/' ) );
    root.pop_back();
    fileFinder.addSearchPath( root );
  }
  fileFinder.addSearchPath( dp::home() + "/media/effects/mdl" );
  fileFinder.addSearchPath( dp::home() + "/media/textures/mdl" );

#define COMPLETE_CONVERSION  1
#if COMPLETE_CONVERSION
  dp::fx::mdl::MaterialBuilder materialBuilder( dp::home() + "/media/dpfx/mdl/MDL.cfg" );

  GLSLBuilder glslBuilder;
#else
  XMLBuilder xmlBuilder;
#endif

  for ( size_t i = 0; i<files.size(); i++ )
  {
    std::cout << "parsing <" << files[i] << ">" << std::endl;
    std::string fname = dp::util::getFileStem( files[i] );

    TiXmlElement * libraryElement = nullptr;
#if COMPLETE_CONVERSION
    materialBuilder.parseFile( files[i], fileFinder );
    libraryElement = glslBuilder.buildPipelines( materialBuilder.getMaterials() );
#else
    xmlBuilder.parseFile( files[i], fileFinder );
    libraryElement = xmlBuilder.getXMLTree();
#endif

    if ( libraryElement )
    {
      std::string fileOut = dp::util::getFilePath( files[i] ) + "/" + fname + ".xml";
      TiXmlDocument document( fileOut.c_str() );
      document.LinkEndChild( new TiXmlDeclaration( "1.0", "utf-8", "" ) );

      document.LinkEndChild( libraryElement );
      DP_VERIFY( document.SaveFile() );
    }
  }

  return( 0 );
}

