// Copyright (c) 2009-2015, NVIDIA CORPORATION. All rights reserved.
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


#include "CameraAnimator.h"
#include <dp/sg/core/FrustumCamera.h>
#include <dp/sg/core/PerspectiveCamera.h>
#include <dp/sg/core/PipelineData.h>
#include <dp/sg/core/Scene.h>

CameraAnimator::CameraAnimator( QObject * parent )
  : QObject( parent )
  , m_cameraIteration(false)
  , m_cameraIterationIndex(~0)
  , m_cameraIterationPauseDuration(1.0)  // one seconds pause between camera moves
  , m_cameraOrbit(false)
  , m_cameraOrbitAxis(0)
  , m_cameraMoveDuration(3.0)            // three seconds per camera move
  , m_cameraMoveTo(false)
  , m_timerID(0)
{
}

CameraAnimator::~CameraAnimator()
{
}

void
CameraAnimator::setViewState( dp::sg::ui::ViewStateSharedPtr const& vsp )
{
  m_viewState = vsp;
}

void
CameraAnimator::cancel()
{
  // set everything to false
  setCameraOrbit( false );
  setCameraMoveTo( false );
  setCameraIteration( false );
}

// start the timer if needed, kill the timer if allowed
void CameraAnimator::startStopTimer( bool &currentFlag, bool newFlag, bool killIt )
{
  if ( currentFlag != newFlag )
  {
    if ( newFlag != ( m_timerID != 0 ) )   // timer needs to be started or killed
    {
      if ( newFlag )
      {
        if ( ! m_timerID )
        {
          m_timerID = startTimer(0);
        }
      }
      else if ( killIt )
      {
        killTimer( m_timerID );
        m_timerID = 0;
      }
    }
    currentFlag = newFlag;
  }
}

double clampDeviation( float start, float end, float part, double minTime )
{
  float f = ( start <= end ) ? end / start : start / end;
  return( dp::math::clamp( ((1.0 - f) / part), minTime, 1.0 ) );
}

void CameraAnimator::orbitCamera( unsigned int axisID, bool cameraRelative, float radians )
{
  dp::sg::core::FrustumCameraSharedPtr const& fch = m_viewState->getCamera().staticCast<dp::sg::core::FrustumCamera>();
  float targetDistance = m_viewState->getTargetDistance();
  dp::math::Vec3f axis( (axisID & BIT0) ? 1.0f : 0.0f,
                        (axisID & BIT1) ? 1.0f : 0.0f,
                        (axisID & BIT2) ? 1.0f : 0.0f );
  normalize( axis );
  fch->orbit( axis, targetDistance, radians, cameraRelative );
}

double CameraAnimator::determineDurationFactor()
{
  // first try: if the start and end orientation differ by more than 90 degree, make the full animation
  dp::math::Quatf diffQuat = m_cameraMoveStart->getOrientation() / m_cameraMoveTarget->getOrientation();
  dp::math::Vec3f axis;
  float angle;
  decompose( diffQuat, axis, angle );
  double durationFactor = dp::math::clamp( (double)(angle / dp::math::PI_HALF), 0.0, 1.0 );

  // second try: if position moves about the focus distance (in fact the medium between start and end), make the full
  // animation.
  float mediumFocus = 0.5f * ( m_cameraMoveStart->getFocusDistance() + m_cameraMoveTarget->getFocusDistance() );
  float posDistance = dp::math::distance( m_cameraMoveStart->getPosition(), m_cameraMoveTarget->getPosition() );
  durationFactor = dp::math::clamp( (double)(posDistance / mediumFocus), durationFactor, 1.0 );

  // third try: if near distance changes by more than 25%, make the full animation
  durationFactor = clampDeviation( m_cameraMoveStart->getNearDistance(), m_cameraMoveTarget->getNearDistance(), 0.25f, durationFactor );

  // fourth try: if focus distance changes by more than 25%, make the full animation
  durationFactor = clampDeviation( m_cameraMoveStart->getFocusDistance(), m_cameraMoveTarget->getFocusDistance(), 0.25f, durationFactor );

  // fifth try: if far distance changes by more than 25%, make the full animation
  durationFactor = clampDeviation( m_cameraMoveStart->getFarDistance(), m_cameraMoveTarget->getFarDistance(), 0.25f, durationFactor );

  // sixth try: if window size changes by more than 25%, make the full animation
  durationFactor = clampDeviation( m_cameraMoveStart->getWindowSize()[0], m_cameraMoveTarget->getWindowSize()[0], 0.25f, durationFactor );
  durationFactor = clampDeviation( m_cameraMoveStart->getWindowSize()[1], m_cameraMoveTarget->getWindowSize()[1], 0.25f, durationFactor );

  // ignore windowOffset, lowerLeft, upperRight for now! I don't expect them to change much.
  return( durationFactor );
}

void CameraAnimator::initCameraMove( dp::sg::core::FrustumCameraSharedPtr const& targetCam )
{
  DP_ASSERT( m_viewState->getCamera().isPtrTo<dp::sg::core::FrustumCamera>() );
  m_cameraMoveStart = m_viewState->getCamera().clone().staticCast<dp::sg::core::FrustumCamera>();
  m_cameraMoveTarget = targetCam->getSharedPtr<dp::sg::core::FrustumCamera>();
  DP_ASSERT( m_cameraMoveStart->getObjectCode() == m_cameraMoveTarget->getObjectCode() );
  cameraMoveDurationFactor( determineDurationFactor() );
}

void CameraAnimator::initCameraMoveToLight( dp::sg::core::LightSourceSharedPtr const& targetLight )
{
  DP_ASSERT( m_viewState->getCamera().isPtrTo<dp::sg::core::FrustumCamera>() );
  m_cameraMoveStart = m_viewState->getCamera().clone().staticCast<dp::sg::core::FrustumCamera>();

  m_cameraMoveTarget = m_cameraMoveStart.clone();

  dp::sg::core::LightSourceSharedPtr lsh( targetLight->getSharedPtr<dp::sg::core::LightSource>() );
  {
    DP_ASSERT( lsh->getLightPipeline() );
    dp::sg::core::PipelineDataSharedPtr const& lp = lsh->getLightPipeline();
    const dp::fx::EffectSpecSharedPtr & es = lp->getEffectSpec();
    for ( dp::fx::EffectSpec::iterator it = es->beginParameterGroupSpecs() ; it != es->endParameterGroupSpecs() ; ++it )
    {
      const dp::sg::core::ParameterGroupDataSharedPtr & parameterGroupData = lp->getParameterGroupData( it );
      if ( parameterGroupData )
      {
        std::string name = (*it)->getName();
        if ( ( name == "standardDirectedLightParameters" )
          || ( name == "standardPointLightParameters" )
          || ( name == "standardSpotLightParameters" ) )
        {
          const dp::fx::ParameterGroupSpecSharedPtr & pgs = parameterGroupData->getParameterGroupSpec();
          if ( name == "standardDirectedLightParameters" )
          {
            m_cameraMoveTarget->setDirection( parameterGroupData->getParameter<dp::math::Vec3f>( pgs->findParameterSpec( "direction" ) ) );
          }
          else if ( name == "standardPointLightParameters" )
          {
            dp::math::Vec3f position = parameterGroupData->getParameter<dp::math::Vec3f>( pgs->findParameterSpec( "position" ) );
            m_cameraMoveTarget->setPosition( position );

            // point us in the direction of the scene center..
            if ( m_viewState->getScene()->getRootNode() )
            {
              dp::math::Vec3f forward = m_viewState->getScene()->getRootNode()->getBoundingSphere().getCenter() - position;
              dp::math::Vec3f worldup( 0.f, 1.f, 0.f ); //pc->getUpVector();

              dp::math::Vec3f right = forward ^ worldup;
              dp::math::Vec3f up = right ^ forward;

              normalize( forward );
              normalize( right );
              normalize( up );

              // X east, Y up, -Z north
              dp::math::Mat33f lookat( {   right[0],    right[1],    right[2],
                                              up[0],       up[1],       up[2],
                                        -forward[0], -forward[1], -forward[2] } );

              dp::math::Quatf ori( lookat );
              m_cameraMoveTarget->setOrientation( ori );
            }
          }
          else
          {
            m_cameraMoveTarget->setPosition( parameterGroupData->getParameter<dp::math::Vec3f>( pgs->findParameterSpec( "position" ) ) );
            m_cameraMoveTarget->setDirection( parameterGroupData->getParameter<dp::math::Vec3f>( pgs->findParameterSpec( "direction" ) ) );
          }
          break;
        }
      }
    }
  }

  cameraMoveDurationFactor( determineDurationFactor() );
}

void CameraAnimator::initCameraZoomAll()
{
  m_cameraMoveStart = m_viewState->getCamera().clone().staticCast<dp::sg::core::FrustumCamera>();
  m_cameraMoveTarget = m_cameraMoveStart.clone();

  if ( m_viewState->getScene()->getRootNode() )
  {
    const dp::math::Sphere3f & bs = m_viewState->getScene()->getRootNode()->getBoundingSphere();
    if ( isPositive( bs ) )
    {
      m_cameraMoveTarget->zoom( bs, dp::math::PI_QUARTER );
      m_viewState->setTargetDistance( std::max<float>( dp::math::distance( m_cameraMoveTarget->getPosition(), bs.getCenter() ), 0.1f ) );
    }
  }

  cameraMoveDurationFactor( determineDurationFactor() );
}

void CameraAnimator::moveCamera( double t )
{
  dp::math::Quatf orientation;
  dp::math::Vec3f position;
  float nearDistance, focusDistance, farDistance;
  dp::math::Vec2f windowOffset, windowSize, lowerLeft, upperRight;
  {
    dp::math::lerp( t, m_cameraMoveStart->getOrientation(), m_cameraMoveTarget->getOrientation(), orientation );
    dp::math::lerp( t, m_cameraMoveStart->getPosition(), m_cameraMoveTarget->getPosition(), position );
    dp::math::lerp( t, m_cameraMoveStart->getNearDistance(), m_cameraMoveTarget->getNearDistance(), nearDistance );
    dp::math::lerp( t, m_cameraMoveStart->getFocusDistance(), m_cameraMoveTarget->getFocusDistance(), focusDistance );
    dp::math::lerp( t, m_cameraMoveStart->getFarDistance(), m_cameraMoveTarget->getFarDistance(), farDistance );
    dp::math::lerp( t, m_cameraMoveStart->getWindowOffset(), m_cameraMoveTarget->getWindowOffset(), windowOffset );
    dp::math::lerp( t, m_cameraMoveStart->getWindowSize(), m_cameraMoveTarget->getWindowSize(), windowSize );
    const dp::math::Box2f & windowRegionStart = m_cameraMoveStart->getWindowRegion();
    const dp::math::Box2f & windowRegionTarget = m_cameraMoveTarget->getWindowRegion();
    dp::math::lerp( t, windowRegionStart.getLower(), windowRegionTarget.getLower(), lowerLeft );
    dp::math::lerp( t, windowRegionStart.getUpper(), windowRegionTarget.getUpper(), upperRight );
  }

  DP_ASSERT( m_viewState->getCamera().isPtrTo<dp::sg::core::FrustumCamera>() );
  dp::sg::core::FrustumCameraSharedPtr const& fch = m_viewState->getCamera().staticCast<dp::sg::core::FrustumCamera>();
  fch->setOrientation( orientation );
  fch->setPosition( position );
  fch->setNearDistance( nearDistance );
  fch->setFocusDistance( focusDistance );
  fch->setFarDistance( farDistance );
  fch->setWindowOffset( windowOffset );
  fch->setWindowSize( windowSize );
  fch->setWindowRegion( dp::math::Box2f( lowerLeft, upperRight ) );
}

void CameraAnimator::setCameraOrbit( bool co )
{
  if ( co )
  {
    // stop move-to and iteration, when orbit starts
    setCameraMoveTo( false );
    setCameraIteration( false );
  }
  startStopTimer( m_cameraOrbit, co, true );
}

void CameraAnimator::setCameraMoveTo( bool cmt )
{
  if ( cmt )
  {
    // stop orbit when move-to starts; iteration is kept unchanged
    setCameraOrbit( false );
  }
  startStopTimer( m_cameraMoveTo, cmt, ! ( m_cameraIteration ) );
  if ( cmt )
  {
    m_cameraMoveTimer.restart();
  }
  else
  {
    m_cameraMoveTimer.stop();
  }
}

// this slot is signaled by the VRW when it determines the distance between current cam and next cam
void CameraAnimator::cameraMoveDurationFactor( double df )
{
  if ( m_cameraMoveDurationFactor < df )
  {
    m_cameraMoveDurationFactor = df;
  }
}

void CameraAnimator::cameraOrbitX( bool tf )
{
  if( tf )
  {
    m_cameraOrbitAxis |= BIT0;
  }
  else
  {
    m_cameraOrbitAxis &= ~BIT0;
  }

  setCameraOrbit( (m_cameraOrbitAxis > 0) );
}

void CameraAnimator::cameraOrbitY( bool tf )
{
  if( tf )
  {
    m_cameraOrbitAxis |= BIT1;
  }
  else
  {
    m_cameraOrbitAxis &= ~BIT1;
  }
  setCameraOrbit( (m_cameraOrbitAxis > 0) );
}

void CameraAnimator::cameraOrbitZ( bool tf )
{
  if( tf )
  {
    m_cameraOrbitAxis |= BIT2;
  }
  else
  {
    m_cameraOrbitAxis &= ~BIT2;
  }
  setCameraOrbit( (m_cameraOrbitAxis > 0) );
}

void CameraAnimator::moveToCamera( dp::sg::core::FrustumCameraSharedPtr const& cam )
{
  m_cameraMoveDurationFactor = 0.0;
  initCameraMove( cam );
  setCameraMoveTo( true );
}

void CameraAnimator::moveToLight( dp::sg::core::LightSourceSharedPtr const& light )
{
  m_cameraMoveDurationFactor = 0.0;
  initCameraMoveToLight( light );
  setCameraMoveTo( true );
}

void CameraAnimator::zoomAll()
{
  m_cameraMoveDurationFactor = 0.0;
  initCameraZoomAll();
  setCameraMoveTo( true );
}

dp::sg::core::FrustumCameraSharedPtr const& CameraAnimator::findNextIterationCamera()
{
  unsigned int noc = m_viewState->getScene()->getNumberOfCameras();

  dp::sg::core::Scene::CameraIterator scci = m_viewState->getScene()->beginCameras();
  unsigned int first = m_cameraIterationIndex++;
  if( m_cameraIterationIndex >= noc )
  {
    m_cameraIterationIndex = 0;
    if ( first >= noc )   // can happen, if the latest camera in iteration has been removed while animating!
    {
      first = noc - 1;    // just take the last camera in the scene as the search end condition
    }
  }

  std::advance( scci, m_cameraIterationIndex );

  while( m_cameraIterationIndex != first )
  {
    // check if its a frustum camera, and is not one of the ones being used in any of the viewports (userdata)
    if ( scci->isPtrTo<dp::sg::core::FrustumCamera>() && (*scci)->getUserData() == nullptr )
    {
      return( scci->inplaceCast<dp::sg::core::FrustumCamera>() );
    }

    ++scci;
    ++m_cameraIterationIndex;

    // start over if we have hit the end
    if( scci == m_viewState->getScene()->endCameras() )
    {
      scci = m_viewState->getScene()->beginCameras();
      m_cameraIterationIndex = 0;
    }
  }

  return dp::sg::core::FrustumCameraSharedPtr::null;
}

void CameraAnimator::setCameraIteration( bool ci )
{
  if ( m_cameraIteration != ci )
  {
    if ( ci )
    {
      // re-use to see if we found an appropriate camera
      ci = false;

      // assure, there are at least two FrustumCameras in the scene
      m_cameraIterationIndex = 0;
      dp::sg::core::FrustumCameraSharedPtr const& fsp = findNextIterationCamera();

      if( fsp )
      {
        moveToCamera( fsp );
        m_cameraIteration = true;
        ci = true;
      }
    }

    // if we are turning off camera iteration, or we didn't find a camera, then turn it off
    if( !ci )
    {
      m_cameraIteration = false;
      setCameraMoveTo( false );   // also stop any running move-to
    }
  }
}

void CameraAnimator::cameraCycle( bool enable )
{
  m_cameraMoveDurationFactor = 0.0;
  setCameraIteration( enable );
}

void CameraAnimator::moveToCameraIndex( unsigned int index )
{
  if( index < m_viewState->getScene()->getNumberOfCameras() )
  {
    dp::sg::core::Scene::CameraIterator scci = m_viewState->getScene()->beginCameras();
    std::advance( scci, index );
    DP_ASSERT( scci->isPtrTo<dp::sg::core::FrustumCamera>() );
    moveToCamera( scci->staticCast<dp::sg::core::FrustumCamera>() );
  }
}

void CameraAnimator::timerEvent( QTimerEvent * te )
{
  if ( m_cameraIteration && ! m_cameraMoveTo )
  {
    if ( ! m_cameraIterationTimer.isRunning() )
    {
      m_cameraIterationTimer.restart();
    }
    else
    {
      double currentTime = m_cameraIterationTimer.getTime();
      if ( m_cameraIterationPauseDuration < currentTime )
      {
        m_cameraIterationTimer.stop();

        // move-to next camera
        dp::sg::core::FrustumCameraSharedPtr const& fsp = findNextIterationCamera();
        if( fsp )
        {
          moveToCamera( fsp );
        }
      }
    }
  }
  else if ( m_cameraMoveTo )
  {
    DP_ASSERT( m_cameraMoveTimer.isRunning() );
    double currentTime = m_cameraMoveTimer.getTime();
    double t;
    double realDuration = m_cameraMoveDuration * m_cameraMoveDurationFactor;
    if ( realDuration < currentTime )
    {
      setCameraMoveTo( false );
      t = 1.0;
    }
    else
    {
      t = currentTime / realDuration;
      t = 0.5 * ( sin( -dp::math::PI_HALF + t * dp::math::PI ) + 1.0 );   // smooth-in and -out using simple sine curve
    }

    moveCamera( t );
    emit update();
  }
  else if ( m_cameraOrbit )
  {
    orbitCamera( m_cameraOrbitAxis, true, dp::math::degToRad(1.0f) );
    emit update();
  }
}

