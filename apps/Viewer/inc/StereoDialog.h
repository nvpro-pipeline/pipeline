// Copyright NVIDIA Corporation 2009-2011
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

#include <ViewerRendererWidget.h>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QLabel>

// X11 defines Bool and Qt doesn't like this Bool
#if defined(Bool)
#undef Bool
#endif

class StereoDialog : public QDialog
{
  Q_OBJECT

  public:
    StereoDialog( QWidget * parent = nullptr, ViewerRendererWidget * vrw = nullptr );
    ~StereoDialog();

  public slots:
    void restore();

  protected slots:
    void setStereoEnable( bool state );
    void setAdjustment( bool state );
    void setDistance( double distance );
    void setReversed( bool state );

  private:
    QLabel          * m_adjustementLabel;
    QDoubleSpinBox  * m_adjustmentSpin;

  private:
    ViewerRendererWidget * m_renderer;

    bool  m_currentAdjustment;
    float m_currentFactor;
    float m_currentDistance;

    bool  m_restoreStereoEnable;
    bool  m_restoreAdjustment;
    float m_restoreFactor;
    float m_restoreDistance;
    bool  m_restoreReversed;
};
