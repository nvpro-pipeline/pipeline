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


#pragma once

#include <string>
#include <vector>

#include <test/testfw/core/Test.h>
#include <dp/util/DynamicLibrary.h>

namespace dp
{
  namespace testfw
  {
    namespace core
    {

      enum class TestReturnFlag
      {
          PASSED = 0
        , FAILED = 1
        , INVALID_INPUT = 2
      };

      class TestObject
      {
        friend DPTCORE_API bool loadTest( const std::string & filename, std::vector<TestObject>& testOut );
      
      public:
        typedef Test * (*TestCreateFunc)();

      public:
        DPTCORE_API TestObject( TestCreateFunc testCreator, const std::string& name, const std::string& desc );
        DPTCORE_API TestObject( TestCreateFunc testCreator, const dp::util::DynamicLibrarySharedPtr& dynLib );

      public:
        DPTCORE_API TestObject( const TestObject& rhs );
        DPTCORE_API ~TestObject();


        DPTCORE_API TestReturnFlag run( MeasurementFunctor& mf, const std::vector<std::string>& options = std::vector<std::string>() );

        const std::string& getName(){ return m_name; }
        const std::string& getDescription(){ return m_description; }

      private:
        TestCreateFunc m_testCreator;
        dp::util::DynamicLibrarySharedPtr m_dynLib;
        
        std::string m_name;
        std::string m_description;
      };
    
    } //namespace core
  } //namespace testfw
} //namespace dp
