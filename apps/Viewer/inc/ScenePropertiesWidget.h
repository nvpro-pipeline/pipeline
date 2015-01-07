
// Copyright NVIDIA Corporation 2009-2010
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

#include <dp/sg/core/Object.h>
#include <QBoxLayout>
#include <QDockWidget>
#include <QValidator>

class ScenePropertiesWidget : public QDockWidget
{
  Q_OBJECT

  public:
    ScenePropertiesWidget(QWidget *parent = 0);
    virtual ~ScenePropertiesWidget();

    virtual void clear();

  public slots:
    void adjustRangesClicked( bool checked );
    void colorButtonClicked( bool checked );
    void currentColorChanged( const QColor & color );
    void currentItemChanged( dp::sg::core::ObjectSharedPtr current, dp::sg::core::ObjectSharedPtr previous );
    void editingFinishedFloat();
    void editingFinishedString();
    void editingFinishedUInt();
    void enumIndexChanged( int index );
    void spinValueChanged( int value );
    void stateChangedBool( int state );
    void textureSelectionClicked( bool checked );
    void valueChangedArray( double value );
    void valueChangedFloat( int value );

  protected:
    void displayItem( dp::sg::core::ObjectSharedPtr const & object );
    void updateItem();

  private:
    class ObjectObserver : public dp::util::Observer
    {
      public:
        ObjectObserver( ScenePropertiesWidget * spw )
          : m_spw(spw)
        {
        }

        virtual void onNotify( const dp::util::Event &event, dp::util::Payload *payload )
        {
          m_spw->updateItem();
        }

        virtual void onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload )
        {
        }

      private:
        ScenePropertiesWidget * m_spw;
    };

  private:
    QWidget * createEdit( bool value, dp::util::PropertyId pid, bool enabled );
    QLayout * createEdit( float value, dp::util::PropertyId pid, bool enabled );
    template<unsigned int N> QLayout * createEdit( const dp::math::Vecnt<N,float> & value, dp::util::PropertyId pid, bool enabled );
    QWidget * createEdit( int value, dp::util::PropertyId pid, bool enabled );
    QWidget * createEdit( unsigned int value, dp::util::PropertyId pid, bool enabled );
    QWidget * createEdit( const std::string & value, dp::util::PropertyId pid, bool enabled );
    QWidget * createEdit( const dp::sg::core::TextureSharedPtr & value, dp::util::PropertyId pid, bool enabled );
    QLayout * createEdit( const dp::math::Quatf & value, dp::util::PropertyId pid, bool enabled );
    QHBoxLayout * createLabledSlider( dp::util::PropertyId pid, float value, float min, float max, unsigned int size = 0, unsigned int index = 0 );
    void updateEdit( QWidget * widget, bool value );
    void updateEdit( QLayout * layout, float value, dp::util::PropertyId pid );
    template<unsigned int N> void updateEdit( QLayout * layout, const dp::math::Vecnt<N,float> & value, dp::util::PropertyId pid );
    void updateEdit( QWidget * widget, int value, dp::util::PropertyId pid );
    void updateEdit( QWidget * widget, unsigned int value );
    void updateEdit( QWidget * widget, const std::string & value );
    void updateEdit( QLayout * layout, const dp::math::Quatf & value );
    void updateEdit( QWidget * widget, const dp::sg::core::TextureSharedPtr & value );
    void updateLabledSlider( QHBoxLayout * layout, float value );

    template<typename T> bool setValue( dp::util::PropertyId pid, const T & value );
    template<typename T> bool setValue( dp::util::PropertyId pid, unsigned int size, unsigned int index, const T & value );
    template<typename T, unsigned int N> bool setSubValue( dp::util::PropertyId pid, unsigned int index, const T & value );

  private:
    QRegExpValidator              * m_hexValidator;
    dp::sg::core::ObjectSharedPtr   m_object;
    ObjectObserver                  m_objectObserver;
};

