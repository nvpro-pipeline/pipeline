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

#include <dp/gl/Config.h>
#include <dp/gl/Object.h>

namespace dp
{
  namespace gl
  {

    class DisplayList;
    typedef dp::util::SmartPtr<DisplayList> SmartDisplayList;

    class DisplayList : public Object
    {
      public:
        /** \brief Create a new shared displaylist
            \return DisplayListSharedPtr to a new displaylist
        **/
        DP_GL_API static SmartDisplayList create();

        DP_GL_API void beginCompile( );
        DP_GL_API void beginCompileAndExecute( );
        DP_GL_API void endCompile( );

        DP_GL_API void execute() const;

      protected:
        /** \brief Construct a new DisplayList.
            \remarks Use \sa compile and \sa compileAndExecute to feed the display list with data **/
        DP_GL_API DisplayList();
        /** \brief Copy construction not allowed, but required for handles. **/
        DP_GL_API DisplayList( const DisplayList & );
        DP_GL_API virtual ~DisplayList();
    };

    inline DisplayList::DisplayList()
    {
      setGLId( glGenLists( 1 ) );
    }

    inline DisplayList::DisplayList( const DisplayList &)
    {
      DP_ASSERT( 0 && "copy constructor may not be called" );
    }

    inline SmartDisplayList DisplayList::create()
    {
      return new DisplayList;
    }

    inline void DisplayList::beginCompile( )
    {
      glNewList( getGLId(), GL_COMPILE );
    }

    inline void DisplayList::beginCompileAndExecute( )
    {
      glNewList( getGLId(), GL_COMPILE_AND_EXECUTE );
    }

    inline void DisplayList::endCompile( )
    {
      glEndList( );
    }

    inline void DisplayList::execute() const
    {
      glCallList( getGLId() );
    }

  } // namespace gl
} // namespace dp
