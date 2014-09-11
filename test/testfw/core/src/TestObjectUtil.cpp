// Copyright NVIDIA Corporation 2011
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


#include <test/testfw/core/TestObjectUtil.h>
#include <dp/util/File.h>
#include <dp/util/DynamicLibrary.h>
#include <dp/util/Tokenizer.h>

#include <iostream>

using namespace std;

namespace dp
{
  namespace testfw
  {
    namespace core
    {

      DPTCORE_API dp::util::SmartDynamicLibrary getTestLib( std::string const& libname )
      {
        static std::map<std::string, dp::util::SmartDynamicLibrary> g_libs;

        std::map<std::string, dp::util::SmartDynamicLibrary>::iterator it = g_libs.find(libname);
        if ( it == g_libs.end() )
        {
          static std::string modulePath;
          if ( modulePath.empty() )
          {
            modulePath = dp::util::getModulePath();
          }

          std::string filePath = modulePath + "/" + libname + ".dptest";
          if ( dp::util::fileExists( filePath ) )
          {
            it = g_libs.insert( std::make_pair( libname, dp::util::DynamicLibrary::createFromFile( filePath ) ) ).first;
          }
          else
          {
            std::cerr << "Error: " << filePath << " could not be found or is not a regular file\n";
            return dp::util::SmartPtr<dp::util::DynamicLibrary>::null;
          }
        }

        return (*it).second;

      }

      DPTCORE_API size_t findTests( const string & path, const string & filter, vector<TestObject>& testsOut )
      {

        size_t wc = filter.find_first_of('/');

        if(wc == string::npos)
        {
          dp::util::SmartDynamicLibrary lib = getTestLib(filter);

          if(!lib)
          {
            return 0;
          }

          TestEnumFunc testEnumerate = (TestEnumFunc) lib->getSymbol("appendTests");

          if ( !testEnumerate )
          {
            cerr << "Error: The library " << filter << " is not a valid test library";
            return 0;
          }

          size_t oldSize = testsOut.size();
          testEnumerate( testsOut );
          
          return testsOut.size() - oldSize;
        }

        if( filter.find_first_of('/', wc+1) != string::npos )
        {
          cerr << "Error: invalid test specification " << filter << "\n";
          return 0;
        }

        util::SmartDynamicLibrary lib = getTestLib( filter.substr(0, wc) );
        TestGetEntryPointFunc getTest = (TestGetEntryPointFunc)lib->getSymbol("getTest");
        TestObject* testObject = getTest( filter.substr(wc+1, filter.length() - wc - 1).c_str() );
        DP_ASSERT( testObject );
        testsOut.push_back( *testObject );

        return 1;
      }
    } //namespace core
  } //namespace testfw
} //namespace dp
