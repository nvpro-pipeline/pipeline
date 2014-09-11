// Copyright NVIDIA Corporation 2002-2005
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
/** \file */

#include <dp/sg/ui/Config.h>
#include <dp/math/Vecnt.h>

namespace dp
{
  namespace sg
  {
    namespace ui
    {

      //! Trackball class for simulating a virtual trackball.
      /** This class simulates a virtual trackball using the last and current mouse position.
        * It simply projects the stuff onto a sphere or hyperbolic sheet and calculates the 
        * axis and angle that is needed to rotate the trackball from the last to the current 
        * position.
        */
      class Trackball
      {
        public:
          //! Default constructor
          DP_SG_UI_API Trackball(void);

          //! Default destructor
          DP_SG_UI_API virtual ~Trackball(void);

          //! Start trackball calculation
          /** Calculate the axis and the angle (radians) 
            * by the given mouse coordinates.
            * Project the points onto the virtual
            * trackball, then figure out the axis of rotation, which is the cross
            * product of p0 p1 and O p0 (O is the center of the ball, 0,0,0)
            * \note This is a deformed trackball-- is a trackball in the center,
            * but is deformed into a hyperbolic sheet of rotation away from the
            * center.
            */
          DP_SG_UI_API void  apply( const dp::math::Vec2f &p0   //!< Last mouse position - components must be scaled to [-1,1]
                              , const dp::math::Vec2f &p1   //!< Current mouse position - components must be scaled to [-1,1]
                              , dp::math::Vec3f & axis      //!< Resulting axis
                              , float & rad       //!< Resulting angle
                              );
      
          //! Set trackball size.
          /** Define the size of the trackball. Default size is 0.8f.
           *  This size should really be based on the distance from the center of
           *  rotation to the point on the object underneath the mouse.  That
           *  point would then track the mouse as closely as possible.
           */
          DP_SG_UI_API void  setSize( float size   //!< %Trackball size
                                );

          //! Determine current size of the trackball.
          /** \return %Trackball size
            */
          DP_SG_UI_API float getSize( void );
      
        protected:
          //! Project the x,y mouse position onto a sphere.
          /** Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
            * if we are away from the center of the sphere.
            */
          DP_SG_UI_API float projectOntoTBSphere( const dp::math::Vec2f & p );

          float m_tbsize;  //!< %Trackball size (default is 0.8f)
      };

      inline  void  Trackball::setSize( float size )
      {
        DP_ASSERT( m_tbsize > 0.f );
        m_tbsize = size;
      }

      inline float Trackball::getSize( void )
      {
        return( m_tbsize );
      }

    } // namespace ui
  } // namespace sg
} //  namespace dp
