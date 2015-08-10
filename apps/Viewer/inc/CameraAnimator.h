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


#pragma once

#include <dp/sg/ui/ViewState.h>
#include <dp/util/Timer.h>
#include <QObject>

class CameraAnimator : public QObject
{
  Q_OBJECT

  Q_PROPERTY(bool cameraOrbitX  READ isCameraOrbitX WRITE cameraOrbitX)
  Q_PROPERTY(bool cameraOrbitY  READ isCameraOrbitX WRITE cameraOrbitX)
  Q_PROPERTY(bool cameraOrbitZ  READ isCameraOrbitX WRITE cameraOrbitX)
  Q_PROPERTY(bool cameraCycle   READ isCameraCycle  WRITE cameraCycle )

public:
  CameraAnimator(QObject *parent = 0);
  ~CameraAnimator();

  void setViewState( dp::sg::ui::ViewStateSharedPtr const& vsp );
  void moveToCamera( dp::sg::core::FrustumCameraSharedPtr const& cam );
  void moveToLight( dp::sg::core::LightSourceSharedPtr const& light );
  void zoomAll();
  void cameraOrbitX( bool );
  void cameraOrbitY( bool );
  void cameraOrbitZ( bool );
  void cameraCycle( bool );
  bool isCameraOrbitX();
  bool isCameraOrbitY();
  bool isCameraOrbitZ();
  bool isCameraCycle();

public slots:
  void cancel();
  void moveToCameraIndex( unsigned int index );

signals:
  void update();

private:
  void cameraMoveDurationFactor(double);
  double determineDurationFactor();
  dp::sg::core::FrustumCameraSharedPtr const& findNextIterationCamera();
  void initCameraMove( dp::sg::core::FrustumCameraSharedPtr const& targetCam );
  void initCameraMoveToLight( dp::sg::core::LightSourceSharedPtr const& targetLight );
  void initCameraZoomAll();
  void moveCamera( double t );
  void orbitCamera( unsigned int axisID, bool cameraRelative, float radians );
  void setCameraIteration( bool cr );
  void setCameraMoveTo( bool cr );
  void setCameraOrbit( bool cr );
  void startStopTimer( bool &currentOpt, bool newOpt, bool dontStop );

  virtual void timerEvent( QTimerEvent * te );

private:
  bool                                  m_cameraIteration;
  unsigned int                          m_cameraIterationIndex;
  double                                m_cameraIterationPauseDuration;
  dp::util::Timer                       m_cameraIterationTimer;
  dp::sg::core::FrustumCameraSharedPtr  m_cameraMoveStart;
  dp::sg::core::FrustumCameraSharedPtr  m_cameraMoveTarget;
  double                                m_cameraMoveDuration;
  double                                m_cameraMoveDurationFactor;
  dp::util::Timer                       m_cameraMoveTimer;
  bool                                  m_cameraMoveTo;
  bool                                  m_cameraOrbit;
  unsigned int                          m_cameraOrbitAxis;
  int                                   m_timerID;
  dp::sg::ui::ViewStateSharedPtr        m_viewState;
};

inline bool CameraAnimator::isCameraOrbitX()
{
  return m_cameraOrbit && (m_cameraOrbitAxis & BIT0);
}

inline bool CameraAnimator::isCameraOrbitY()
{
  return m_cameraOrbit && (m_cameraOrbitAxis & BIT1);
}

inline bool CameraAnimator::isCameraOrbitZ()
{
  return m_cameraOrbit && (m_cameraOrbitAxis & BIT2);
}

inline bool CameraAnimator::isCameraCycle()
{
  return m_cameraIteration;
}

