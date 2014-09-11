// Copyright NVIDIA Corporation 2012
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


#include <test/testfw/core/TestRender.h>
#include <test/rix/core/framework/RiXBackend.h>

#include <dp/rix/core/RiX.h>
#include <dp/rix/gl/RiXGL.h>

class Feature_tangent_space : public dp::testfw::core::TestRender
{
public:
  Feature_tangent_space();
  ~Feature_tangent_space();

  bool onInit( void );
  bool onRun( unsigned int idx );
  bool onClear( void );
  bool onRunCheck( unsigned int i );

private:
  void createScene();

private:

  unsigned int m_numFrames;

  dp::rix::core::ContainerEntry m_containerEntryLightDir;
  dp::rix::core::ContainerSharedHandle m_fragConstContainer;

  dp::rix::core::test::framework::RenderDataRiX* m_renderData;
  dp::rix::core::Renderer* m_rix;
};

extern "C"
{
  DPTTEST_API dp::testfw::core::Test * create_feature_tangent_space()
  {
    return new Feature_tangent_space();
  }
}
