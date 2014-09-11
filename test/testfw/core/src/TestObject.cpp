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


#include <test/testfw/core/Test.h>
#include <test/testfw/core/TestObject.h>

#include <dp/util/File.h>
#include <dp/util/DynamicLibrary.h>

#include <dp/util/DPAssert.h>
#include <memory>
#include <iostream>

using namespace std;

namespace dp
{
  namespace testfw
  {
    namespace core
    {

      TestReturnFlag TestObject::run( MeasurementFunctor& mf, const vector<string>& options )
      {
        DP_ASSERT( m_testCreator );

        std::auto_ptr<Test> test( m_testCreator() );

        if( !test->option( options ) || !mf.option( options ) )
        {
          return TRF_INVALID_INPUT;
        }

        return test->run(mf, m_name) ? TRF_PASSED : TRF_FAILED;
      }

      TestObject::TestObject( TestCreateFunc testCreator, const dp::util::SmartDynamicLibrary& dynLib )
        : m_testCreator(testCreator)
        , m_dynLib(dynLib)
      {

      }

      TestObject::TestObject( TestCreateFunc testCreator, const std::string& name, const std::string& desc  )
        : m_testCreator(testCreator)
        , m_name(name)
        , m_description(desc)
      {

      }

      TestObject::TestObject( const TestObject& rhs )
        : m_testCreator(rhs.m_testCreator)
        , m_dynLib(rhs.m_dynLib)
        , m_name(rhs.m_name)
        , m_description(rhs.m_description)
      {
      }

      TestObject::~TestObject()
      {
      }

    } //namespace core
  } //namespace testfw
} //namespace dp
