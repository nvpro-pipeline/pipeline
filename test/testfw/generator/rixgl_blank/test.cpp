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


#include <test/testfw/manager/Manager.h>
#include "##test##.h"

#include <dp/math/Trafo.h>

#include <test/rix/core/framework/RiXBackend.h>
#include <test/rix/core/helpers/GeometryHelper.h>

#include <limits>

//Automatically add the test to the module's global test list
REGISTER_TEST("##test##", "##description##", create_##test##);

using namespace dp;
using namespace rix::core;

##test##::##test##()
{
}

##test##::~##test##()
{
}

bool ##test##::onInit()
{
  DP_ASSERT( dynamic_cast<test::framework::RiXBackend*>(&(*m_backend)) );
  m_rix = static_cast<test::framework::RiXBackend*>(&(*m_backend))->getRenderer();
  m_renderData = new test::framework::RenderDataRiX;
  util::smart_cast<util::gl::RenderTargetGL>( m_displayTarget )->setClearColor( 0.46f, 0.72f, 0.0f, 1.0f );

  createScene();

  return true;  
}

bool ##test##::onRun( unsigned int idx )
{
  render(m_renderData, m_displayTarget);

  return true;
}

bool ##test##::onClear()
{
  delete m_renderData;

  return true;
}

void ##test##::createScene()
{
  SmartRenderGroupHandle renderGroup = m_rix->renderGroupCreate();
  m_renderData->setRenderGroup(renderGroup);
}
//##test##