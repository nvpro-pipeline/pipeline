// Copyright (c) 2009-2016, NVIDIA CORPORATION. All rights reserved.
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


#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QScrollArea>
#include <QSlider>
#include <QSpinBox>
#include <QVariant>
#include "ScenePropertiesWidget.h"
#include "Viewer.h"
#include <dp/sg/core/ParameterGroupData.h>
#include <dp/sg/core/TextureFile.h>
#include <dp/sg/io/IO.h>

float getRealValue( QSlider * slider )
{
  DP_ASSERT( slider->minimum() == 0 );
  return( dp::math::lerp( (float)slider->value() / slider->maximum()
                        , slider->property( "Min" ).toFloat()
                        , slider->property( "Max" ).toFloat() ) );
}

void setRealValue( QSlider * slider, float value )
{
  DP_ASSERT( slider->minimum() == 0 );
  float min = slider->property( "Min" ).toFloat();
  float max = slider->property( "Max" ).toFloat();
  slider->setValue( slider->maximum() * ( value - min ) / ( max - min ) );
}

bool isHiddenProperty( dp::util::PropertyId pid )
{
  std::string annotation = pid->getAnnotation();
  return( annotation.find( "_anno_hidden" ) != std::string::npos );
}

std::string getAnnotation( dp::util::PropertyId pid, const std::string & name )
{
  std::string annotation = pid->getAnnotation();
  size_t startPos = annotation.find( name );
  if ( startPos != std::string::npos )
  {
    startPos = annotation.find( '(', startPos );
    DP_ASSERT( startPos != std::string::npos );
    size_t endPos = annotation.find( ')', startPos );
    DP_ASSERT( endPos != std::string::npos );
    return( annotation.substr( startPos + 1, endPos - startPos - 1 ) );
  }
  return( "" );
}

std::string getDisplayName( dp::util::PropertyId pid, const std::string & propertyName )
{
  std::string displayName = getAnnotation( pid, "_anno_displayName" );
  return( displayName.empty() ? propertyName : displayName );
}

template<typename T>
bool isRanged( dp::util::PropertyId pid, const std::string & name, T & min, T & max )
{
  std::string range = getAnnotation( pid, name );
  if ( ! range.empty() )
  {
    std::istringstream iss( range );
    char comma;
    iss >> min >> comma >> max;
    return( true );
  }
  return( false );
}

static const char baseTitle[] = "Object Properties";

ScenePropertiesWidget::ScenePropertiesWidget(QWidget *parent)
  : QDockWidget( baseTitle, parent )
  , m_hexValidator(nullptr)
  , m_objectObserver(this)
{
  setObjectName( "ScenePropertiesWidget" );
}

ScenePropertiesWidget::~ScenePropertiesWidget()
{
  if ( m_object )
  {
    m_object->detach( &m_objectObserver );
  }
}

template<typename T>
bool ScenePropertiesWidget::setValue( dp::util::PropertyId pid, const T & value )
{
  bool done = false;
  if ( value != m_object->getValue<T>( pid ) )
  {
    m_object->setValue( pid, value );
    GetApp()->emitMaterialChanged();
    done = true;
  }
  return( done );
}

template<typename T, unsigned int N>
bool ScenePropertiesWidget::setSubValue( dp::util::PropertyId pid, unsigned int index, const T & value )
{
  dp::math::Vecnt<N,T> completeValue = m_object->getValue<dp::math::Vecnt<N,T> >( pid );
  if ( value != completeValue[index] )
  {
    completeValue[index] = value;
    m_object->setValue( pid, completeValue );
    return( true );
  }
  return( false );
}

template<typename T>
bool ScenePropertiesWidget::setValue( dp::util::PropertyId pid, unsigned int size, unsigned int index, const T & value )
{
  bool done = false;
  switch( size )
  {
    case 2 :
      done = setSubValue<T,2>( pid, index, value );
      break;
    case 3 :
      done = setSubValue<T,3>( pid, index, value );
      break;
    case 4 :
      done = setSubValue<T,4>( pid, index, value );
      break;
    default :
      DP_ASSERT( false );
      break;
  }
  if ( done )
  {
    GetApp()->emitMaterialChanged();
  }
  return( done );
}

void ScenePropertiesWidget::adjustRangesClicked( bool checked )
{
  DP_ASSERT( dynamic_cast<QPushButton*>(sender()) );
  QPushButton * button = static_cast<QPushButton*>(sender());

  QHBoxLayout * layout = static_cast<QHBoxLayout *>( button->property( "Layout" ).value<void *>() );
  DP_ASSERT( 1 < layout->count() );
  QLayoutItem * item = layout->itemAt( 0 );
  DP_ASSERT( item->layout() && dynamic_cast<QHBoxLayout*>(item->layout()) );
  QHBoxLayout * labledSliderLayout = static_cast<QHBoxLayout*>(item->layout());

  DP_ASSERT( labledSliderLayout->count() == 2 );
  item = labledSliderLayout->itemAt( 1 );
  DP_ASSERT( item->widget() && dynamic_cast<QSlider*>(item->widget()) );

  QSlider * slider = static_cast<QSlider*>(item->widget());
  float min = slider->property( "Min" ).toFloat();
  float max = slider->property( "Max" ).toFloat();

  QLineEdit * minEdit = new QLineEdit( QString( "%1" ).arg( min ) );
  minEdit->setValidator( new QDoubleValidator() );

  QLineEdit * maxEdit = new QLineEdit( QString( "%1" ).arg( max ) );
  maxEdit->setValidator( new QDoubleValidator() );

  QFrame * separatorLine = new QFrame;
  separatorLine->setFrameShape( QFrame::HLine );
  separatorLine->setFrameShadow( QFrame::Sunken );

  QDialogButtonBox * buttonBox = new QDialogButtonBox( QDialogButtonBox::Ok | QDialogButtonBox::Cancel );

  QFormLayout * formLayout = new QFormLayout();
  formLayout->addRow( "Minimum", minEdit );
  formLayout->addRow( "Maximum", maxEdit );
  formLayout->addRow( separatorLine );
  formLayout->addRow( buttonBox );

  QDialog dialog;
  dialog.setWindowTitle( "Adjust Ranges" );
  dialog.setLayout( formLayout );

  connect( buttonBox, SIGNAL(accepted()), &dialog, SLOT(accept()) );
  connect( buttonBox, SIGNAL(rejected()), &dialog, SLOT(reject()) );

  if ( dialog.exec() == QDialog::Accepted )
  {
    float newMin = minEdit->text().toFloat();
    float newMax = maxEdit->text().toFloat();
    if ( newMin < newMax )
    {
      if ( ( newMin != min ) || ( newMax != max ) )
      {
        float currentValue = getRealValue( slider );
        slider->setProperty( "Min", newMin );
        slider->setProperty( "Max", newMax );
        setRealValue( slider, currentValue );

        DP_ASSERT( labledSliderLayout->itemAt( 0 ) && dynamic_cast<QLabel*>(labledSliderLayout->itemAt( 0 )->widget()) );
        QLabel * label = static_cast<QLabel*>(labledSliderLayout->itemAt( 0 )->widget());
        int fieldWidth = label->property( "FieldWidth" ).toInt();
        int precision = label->property( "Precision" ).toInt();
        label->setText( QString( "%1" ).arg( getRealValue( slider ), fieldWidth, 'f', precision ) );
      }
    }
    else
    {
      QMessageBox box;
      box.setWindowTitle( "Adjust Ranges" );
      box.setText( "The minimum value has to be less than the maximum value" );
      box.exec();
    }
  }
}

void ScenePropertiesWidget::colorButtonClicked( bool checked )
{
  DP_ASSERT( dynamic_cast<QPushButton*>(sender()) );
  QPushButton * button = static_cast<QPushButton*>(sender());

  dp::util::PropertyId pid = static_cast<dp::util::PropertyId>( button->property( "dp::util::PropertyId" ).value<void *>() );
  unsigned int channels = button->property( "Channels" ).toUInt();
  DP_ASSERT( ( channels == 3 ) || ( channels == 4 ) );

  QString currentColor = button->text();
  QColorDialog colorDialog( currentColor, this );
  colorDialog.setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
  colorDialog.setProperty( "Channels", channels );
  if ( channels == 4 )
  {
    colorDialog.setOption( QColorDialog::ShowAlphaChannel );
  }
  connect( &colorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(currentColorChanged(const QColor&)) );

  if ( colorDialog.exec() == QDialog::Accepted )
  {
    QColor color = colorDialog.selectedColor();
    button->setStyleSheet( QString( "QPushButton { color:" ) + color.name() + QString( "; }" ) );
    button->setText( color.name() );
  }
  else
  {
    QColor color( currentColor );
    if ( channels == 3 )
    {
      setValue<dp::math::Vec3f>( pid, dp::math::Vec3f( color.redF(), color.greenF(), color.blueF() ) );
    }
    else
    {
      setValue<dp::math::Vec4f>( pid, dp::math::Vec4f( color.redF(), color.greenF(), color.blueF(), color.alphaF() ) );
    }
  }
}

void ScenePropertiesWidget::currentColorChanged( const QColor & color )
{
  dp::util::PropertyId pid = static_cast<dp::util::PropertyId>( sender()->property( "dp::util::PropertyId" ).value<void *>() );
  unsigned int channels = sender()->property( "Channels" ).toUInt();

  if ( channels == 3 )
  {
    DP_VERIFY( setValue<dp::math::Vec3f>( pid, dp::math::Vec3f( color.redF(), color.greenF(), color.blueF() ) ) );
  }
  else
  {
    DP_VERIFY( setValue<dp::math::Vec4f>( pid, dp::math::Vec4f( color.redF(), color.greenF(), color.blueF(), color.alphaF() ) ) );
  }
}

void ScenePropertiesWidget::editingFinishedFloat()
{
  DP_ASSERT( dynamic_cast<QLineEdit*>(sender()) );

  dp::util::PropertyId pid = static_cast<dp::util::PropertyId>( sender()->property( "dp::util::PropertyId" ).value<void *>() );
  float value = static_cast<QLineEdit*>(sender())->text().toFloat();
  setValue<float>( pid, value );
}

void ScenePropertiesWidget::editingFinishedString()
{
  dp::util::PropertyId pid = static_cast<dp::util::PropertyId>( sender()->property( "dp::util::PropertyId" ).value<void *>() );

  DP_ASSERT( dynamic_cast<QLineEdit*>(sender()) );
  QLineEdit * lineEdit = static_cast<QLineEdit*>(sender());
  std::string value = lineEdit->text().toStdString();

  setValue<std::string>( pid, value );
}

void ScenePropertiesWidget::editingFinishedUInt()
{
  dp::util::PropertyId pid = static_cast<dp::util::PropertyId>( sender()->property( "dp::util::PropertyId" ).value<void *>() );

  DP_ASSERT( dynamic_cast<QLineEdit*>(sender()) );
  QLineEdit * lineEdit = static_cast<QLineEdit*>(sender());

  bool ok;
  unsigned int value = lineEdit->text().toUInt( &ok, 16 );
  DP_ASSERT( ok );

  setValue<unsigned int>( pid, value );
}

void ScenePropertiesWidget::enumIndexChanged( int index )
{
  dp::util::PropertyId pid = static_cast<dp::util::PropertyId>( sender()->property( "dp::util::PropertyId" ).value<void *>() );
  DP_ASSERT( pid->isEnum() );
  DP_VERIFY( setValue<int>( pid, index ) );
}

void ScenePropertiesWidget::spinValueChanged( int value )
{
  dp::util::PropertyId pid = static_cast<dp::util::PropertyId>( sender()->property( "dp::util::PropertyId" ).value<void *>() );
  DP_VERIFY( setValue<int>( pid, value ) );
}

void ScenePropertiesWidget::stateChangedBool( int state )
{
  DP_ASSERT( state != Qt::PartiallyChecked );

  dp::util::PropertyId pid = static_cast<dp::util::PropertyId>( sender()->property( "dp::util::PropertyId" ).value<void *>() );

  DP_VERIFY( setValue<bool>( pid, state == Qt::Checked ) );
}

void ScenePropertiesWidget::textureSelectionClicked( bool checked )
{
  DP_ASSERT( m_object.isPtrTo<dp::sg::core::Sampler>() );
  dp::sg::core::SamplerSharedPtr const& sampler = m_object.staticCast<dp::sg::core::Sampler>();

  QString textureFile = GetApp()->getMainWindow()->getTextureFile( textureTargetToType( sampler->getTexture()->getTextureTarget() ) );
  if ( ! textureFile.isEmpty() )
  {
    DP_ASSERT( dynamic_cast<QPushButton*>(sender()) );
    QPushButton * button = static_cast<QPushButton*>(sender());
    dp::util::PropertyId pid = static_cast<dp::util::PropertyId>( button->property( "dp::util::PropertyId" ).value<void *>() );
    setValue<dp::sg::core::TextureSharedPtr>( pid, dp::sg::io::loadTextureHost( textureFile.toStdString() ) );
    button->setText( textureFile );
  }
}

void ScenePropertiesWidget::valueChangedArray( double value )
{
  dp::util::PropertyId pid = static_cast<dp::util::PropertyId>( sender()->property( "dp::util::PropertyId" ).value<void *>() );
  unsigned int channels = sender()->property( "Channels" ).toUInt();
  unsigned int index = sender()->property( "Index" ).toUInt();
  DP_ASSERT( index < channels );

  setValue<float>( pid, channels, index, value );
}

void ScenePropertiesWidget::valueChangedFloat( int value )
{
  DP_ASSERT( dynamic_cast<QSlider*>(sender()) );
  QSlider * slider = static_cast<QSlider*>(sender());

  float newValue = getRealValue( slider );

  QLabel * label = static_cast<QLabel*>( slider->property( "Label" ).value<void*>() );
  int fieldWidth = label->property( "FieldWidth" ).toInt();
  int precision = label->property( "Precision" ).toInt();

  label->setText( QString( "%1" ).arg( newValue, fieldWidth, 'f', precision ) );

  unsigned int size = slider->property( "Size" ).toUInt();
  dp::util::PropertyId pid = static_cast<dp::util::PropertyId>( slider->property( "dp::util::PropertyId" ).value<void *>() );
  if ( size )
  {
    setValue( pid, size, slider->property( "Index" ).toUInt(), newValue );
  }
  else
  {
    setValue<float>( pid, newValue );
  }
}

QWidget * ScenePropertiesWidget::createEdit( bool value, dp::util::PropertyId pid, bool enabled )
{
  DP_ASSERT( enabled );
  QCheckBox * checkBox = new QCheckBox();
  checkBox->setChecked( value );
  checkBox->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
  connect( checkBox, SIGNAL(stateChanged(int)), this, SLOT(stateChangedBool(int)) );

  return( checkBox );
}

void ScenePropertiesWidget::updateEdit( QWidget * widget, bool value )
{
  DP_ASSERT( dynamic_cast<QCheckBox*>(widget) );
  static_cast<QCheckBox*>(widget)->setChecked( value );
}

QLayout * ScenePropertiesWidget::createEdit( float value, dp::util::PropertyId pid, bool enabled )
{
  DP_ASSERT( enabled );
  float min, max;
  if ( isRanged( pid, "_anno_hardRange", min, max ) )
  {
    return( createLabledSlider( pid, value, min, max ) );
  }
  else if ( isRanged( pid, "_anno_softRange", min, max ) )
  {
    QHBoxLayout * labledSlider = createLabledSlider( pid, value, min, max );

    QPushButton * adjustRangesButton = new QPushButton( "[.]" );
    connect( adjustRangesButton, SIGNAL(clicked(bool)), this, SLOT(adjustRangesClicked(bool)) );

    QHBoxLayout * layout = new QHBoxLayout();
    layout->addLayout( labledSlider );
    layout->addWidget( adjustRangesButton );
    layout->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );

    adjustRangesButton->setProperty( "Layout", QVariant::fromValue( static_cast<void *>( layout ) ) );

    return( layout );
  }
  else
  {
    QLineEdit * lineEdit = new QLineEdit( QString( "%1" ).arg( value ) );
    lineEdit->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
    lineEdit->setValidator( new QDoubleValidator() );
    connect( lineEdit, SIGNAL(editingFinished()), this, SLOT(editingFinishedFloat()) );

    QHBoxLayout * layout = new QHBoxLayout();
    layout->addWidget( lineEdit );
    layout->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );

    return( layout );
  }
}

void ScenePropertiesWidget::updateEdit( QLayout * layout, float value, dp::util::PropertyId pid )
{
  DP_ASSERT( layout );
  float min, max;
  if ( isRanged( pid, "_anno_hardRange", min, max ) )
  {
    DP_ASSERT( dynamic_cast<QHBoxLayout*>(layout) );
    DP_ASSERT( ( min <= value ) && ( value <= max ) );
    updateLabledSlider( static_cast<QHBoxLayout*>(layout), value );
  }
  else if ( isRanged( pid, "_anno_softRange", min, max ) )
  {
    DP_ASSERT( layout->itemAt( 0 ) && dynamic_cast<QHBoxLayout*>(layout->itemAt( 0 )->layout()) );
    updateLabledSlider( static_cast<QHBoxLayout*>(layout->itemAt( 0 )->layout()), value );
  }
  else
  {
    DP_ASSERT( layout->itemAt( 0 ) && dynamic_cast<QLineEdit*>(layout->itemAt( 0 )->widget()) );
    QLineEdit * lineEdit = static_cast<QLineEdit*>(layout->itemAt( 0 )->widget());
    lineEdit->setText( QString( "%1" ).arg( value ) );
  }
}

template<unsigned int N>
QLayout * ScenePropertiesWidget::createEdit( const dp::math::Vecnt<N,float> & value, dp::util::PropertyId pid, bool enabled )
{
  float min, max;
  if ( isRanged( pid, "_anno_hardRange", min, max ) )
  {
    DP_ASSERT( enabled );
    QVBoxLayout * sliderLayout = new QVBoxLayout();
    for ( unsigned int i=0 ; i<N ; i++ )
    {
      sliderLayout->addLayout( createLabledSlider( pid, value[i], min, max, N, i ) );
    }
    sliderLayout->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
    return( sliderLayout );
  }
  else if ( isRanged( pid, "_anno_softRange", min, max ) )
  {
    DP_ASSERT( enabled );
    QPushButton * adjustRangesButton = new QPushButton( "[.]" );
    connect( adjustRangesButton, SIGNAL(clicked(bool)), this, SLOT(adjustRangesClicked(bool)) );

    QVBoxLayout * sliderLayout = new QVBoxLayout();
    for ( unsigned int i=0 ; i<N ; i++ )
    {
      sliderLayout->addLayout( createLabledSlider( pid, value[i], min, max, N, i ) );
    }

    QHBoxLayout * layout = new QHBoxLayout();
    layout->addLayout( sliderLayout );
    layout->addWidget( adjustRangesButton );
    layout->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );

    adjustRangesButton->setProperty( "Layout", QVariant::fromValue( static_cast<void *>( sliderLayout ) ) );

    return( layout );
  }
  else if ( pid->getSemantic() == dp::util::Semantic::COLOR )
  {
    DP_ASSERT( enabled );
    DP_ASSERT( ( N == 3 ) || ( N == 4 ) );
    DP_ASSERT( ( 0.0f <= value[0] ) && ( value[0] <= 1.0f ) );
    DP_ASSERT( ( 0.0f <= value[1] ) && ( value[1] <= 1.0f ) );
    DP_ASSERT( ( 0.0f <= value[2] ) && ( value[2] <= 1.0f ) );
    QColor color( int( 255 * value[0] ), int( 255 * value[1] ), int( 255 * value[2] ) );
    if ( N == 4 )
    {
      DP_ASSERT( ( 0.0f <= value[3] ) && ( value[3] <= 1.0f ) );
      color.setAlphaF( value[3] );
    }

    QPushButton * button = new QPushButton( color.name() );
    button->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
    button->setProperty( "Channels", N );
    button->setStyleSheet( QString( "QPushButton { color:" ) + color.name() + QString( "; }" ) );
    button->setAutoFillBackground( true );

    connect( button, SIGNAL(clicked(bool)), this, SLOT(colorButtonClicked(bool)) );

    QHBoxLayout * layout = new QHBoxLayout();
    layout->addWidget( button );
    layout->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );

    return( layout );
  }
  else
  {
    QHBoxLayout * layout = new QHBoxLayout();
    for ( unsigned int i=0 ; i<N ; i++ )
    {
      if ( enabled )
      {
        QDoubleSpinBox * spinBox = new QDoubleSpinBox();
        spinBox->setRange( -std::numeric_limits<float>::max(), std::numeric_limits<float>::max() );
        spinBox->setSingleStep( 0.01 );
        spinBox->setValue( value[i] );
        spinBox->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
        spinBox->setProperty( "Channels", N );
        spinBox->setProperty( "Index", i );
        connect( spinBox, SIGNAL(valueChanged(double)), this, SLOT(valueChangedArray(double)) );
        layout->addWidget( spinBox );
      }
      else
      {
        layout->addWidget( new QLabel( QString( "%1" ).arg( value[i] ) ) );
      }
    }
    layout->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );

    return( layout );
  }
}

template<unsigned int N>
void ScenePropertiesWidget::updateEdit( QLayout * layout, const dp::math::Vecnt<N,float> & value, dp::util::PropertyId pid )
{
  float min, max;
  if ( isRanged( pid, "_anno_hardRange", min, max ) )
  {
    for ( unsigned int i=0 ; i<N ; i++ )
    {
      DP_ASSERT( layout->itemAt( i ) && dynamic_cast<QHBoxLayout*>(layout->itemAt( i )->layout()) );
      updateLabledSlider( static_cast<QHBoxLayout*>(layout->itemAt( i )->layout() ), value[i] );
    }
  }
  else if ( isRanged( pid, "_anno_softRange", min, max ) )
  {
    DP_ASSERT( layout->itemAt( 0 ) && dynamic_cast<QVBoxLayout*>(layout->itemAt( 0 )->layout()) );
    QLayout * sliderLayout = static_cast<QVBoxLayout*>(layout->itemAt( 0 )->layout());
    for ( unsigned int i=0 ; i<N ; i++ )
    {
      DP_ASSERT( sliderLayout->itemAt( i ) && dynamic_cast<QHBoxLayout*>(sliderLayout->itemAt( i )->layout()) );
      updateLabledSlider( static_cast<QHBoxLayout*>(sliderLayout->itemAt( i )->layout() ), value[i] );
    }
  }
  else if ( pid->getSemantic() == dp::util::Semantic::COLOR )
  {
    DP_ASSERT( ( N == 3 ) || ( N == 4 ) );
    DP_ASSERT( ( 0.0f <= value[0] ) && ( value[0] <= 1.0f ) );
    DP_ASSERT( ( 0.0f <= value[1] ) && ( value[1] <= 1.0f ) );
    DP_ASSERT( ( 0.0f <= value[2] ) && ( value[2] <= 1.0f ) );
    QColor color( int( 255 * value[0] ), int( 255 * value[1] ), int( 255 * value[2] ) );
    if ( N == 4 )
    {
      DP_ASSERT( ( 0.0f <= value[3] ) && ( value[3] <= 1.0f ) );
      color.setAlphaF( value[3] );
    }

    DP_ASSERT( layout->itemAt( 0 ) && dynamic_cast<QPushButton*>(layout->itemAt( 0 )->widget()) );
    QPushButton * button = static_cast<QPushButton*>(layout->itemAt( 0 )->widget());
    button->setText( color.name() );
    button->setStyleSheet( QString( "QPushButton { color:" ) + color.name() + QString( "; }" ) );
  }
  else
  {
    DP_ASSERT( dynamic_cast<QHBoxLayout*>(layout) );
    QHBoxLayout * boxLayout = static_cast<QHBoxLayout*>(layout);
    for ( unsigned int i=0 ; i<N ; i++ )
    {
      DP_ASSERT( boxLayout->itemAt( i ) );
      if ( dynamic_cast<QDoubleSpinBox*>(boxLayout->itemAt( i )->widget() ) )
      {
        static_cast<QDoubleSpinBox*>(boxLayout->itemAt( i )->widget())->setValue( value[i] );
      }
      else
      {
        DP_ASSERT( dynamic_cast<QLabel*>(boxLayout->itemAt( i )->widget()) );
        static_cast<QLabel*>(boxLayout->itemAt( i )->widget())->setText( QString( "%1" ).arg( value[i] ) );
      }
    }
  }
}

QWidget * ScenePropertiesWidget::createEdit( int value, dp::util::PropertyId pid, bool enabled )
{
  DP_ASSERT( enabled );
  int min, max;
  if ( pid->isEnum() )
  {
    QComboBox * comboBox = new QComboBox;
    comboBox->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
    unsigned int numEnums = pid->getEnumsCount();
    for ( unsigned int i=0 ; i<numEnums ; i++ )
    {
      comboBox->addItem( pid->getEnumName( i ).c_str() );
    }
    DP_ASSERT( static_cast<unsigned int>(value) < numEnums );
    comboBox->setCurrentIndex( value );
    connect( comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(enumIndexChanged(int)) );
    return( comboBox );
  }
  else if ( isRanged( pid, "_anno_hardRange", min, max ) )
  {
    DP_ASSERT( min < max );
    DP_ASSERT( ( min <= value ) && ( value <= max ) );

    QComboBox * comboBox = new QComboBox;
    comboBox->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
    for ( int i=min ; i<=max ; i++ )
    {
      comboBox->addItem( QString( "%1" ).arg( i ) );
    }
    comboBox->setCurrentIndex( value - min );
    connect( comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(enumIndexChanged(int)) );
    return( comboBox );
  }
  else
  {
    QSpinBox * spinBox = new QSpinBox();
    spinBox->setValue( value );
    spinBox->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
    connect( spinBox, SIGNAL(valueChanged(int)), this, SLOT(spinValueChanged(int)) );
    return( spinBox );
  }
}

void ScenePropertiesWidget::updateEdit( QWidget * widget, int value, dp::util::PropertyId pid )
{
  DP_ASSERT( widget );
  int min, max;
  if ( pid->isEnum() )
  {
    DP_ASSERT( ( 0 <= value ) && ( (unsigned int)(value) < pid->getEnumsCount() ) );
    DP_ASSERT( dynamic_cast<QComboBox*>(widget) );
    QComboBox * comboBox = static_cast<QComboBox*>(widget);
    comboBox->setCurrentIndex( value );
  }
  else if ( isRanged( pid, "_anno_hardRange", min, max ) )
  {
    DP_ASSERT( ( min <= value ) && ( value <= max ) );
    DP_ASSERT( dynamic_cast<QComboBox*>(widget) );
    QComboBox * comboBox = static_cast<QComboBox*>(widget);
    comboBox->setCurrentIndex( value - min );
  }
  else
  {
    DP_ASSERT( dynamic_cast<QSpinBox*>(widget) );
    QSpinBox * spinBox = static_cast<QSpinBox*>(widget);
    spinBox->setValue( value );
  }
}

QWidget * ScenePropertiesWidget::createEdit( unsigned int value, dp::util::PropertyId pid, bool enabled )
{
  QString text = QString( "0x%1" ).arg( value, 8, 16, (const QChar &)'0' );
  if ( enabled )
  {
    if ( ! m_hexValidator )
    {
      QRegExp regExp( "^0x[0-9|A-F|a-f]{1,8}$" );
      m_hexValidator = new QRegExpValidator( regExp, this );
    }

    QLineEdit * lineEdit = new QLineEdit( text );
    lineEdit->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
    lineEdit->setValidator( m_hexValidator );
    connect( lineEdit, SIGNAL(editingFinished()), this, SLOT(editingFinishedUInt()) );

    return( lineEdit );
  }
  else
  {
    QLabel * label = new QLabel( text );
    label->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
    return( label );
  }
}

void ScenePropertiesWidget::updateEdit( QWidget * widget, unsigned int value )
{
  QString text = QString( "0x%1" ).arg( value, 8, 16, (const QChar &)'0' );
  if ( dynamic_cast<QLineEdit*>(widget) )
  {
    DP_ASSERT( !"never passed this path" );
    QLineEdit * lineEdit = static_cast<QLineEdit*>(widget);
    lineEdit->setText( text );
  }
  else
  {
    DP_ASSERT( dynamic_cast<QLabel*>(widget) );
    static_cast<QLabel*>(widget)->setText( text );
  }
}

QWidget * ScenePropertiesWidget::createEdit( const std::string & value, dp::util::PropertyId pid, bool enabled )
{
  DP_ASSERT( enabled );
  QLineEdit * lineEdit = new QLineEdit( value.c_str() );
  lineEdit->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
  connect( lineEdit, SIGNAL(editingFinished()), this, SLOT(editingFinishedString()) );

  return( lineEdit );
}

void ScenePropertiesWidget::updateEdit( QWidget * widget, const std::string & value )
{
  DP_ASSERT( dynamic_cast<QLineEdit*>(widget) );
  static_cast<QLineEdit*>(widget)->setText( value.c_str() );
}

QWidget * ScenePropertiesWidget::createEdit( const dp::sg::core::TextureSharedPtr & value, dp::util::PropertyId pid, bool enabled )
{
  DP_ASSERT( enabled );
  std::string fileName;
  if ( value.isPtrTo<dp::sg::core::TextureFile>() )
  {
    fileName = value.staticCast<dp::sg::core::TextureFile>()->getFilename();
  }
  else if ( value.isPtrTo<dp::sg::core::TextureHost>() )
  {
    fileName = value.staticCast<dp::sg::core::TextureHost>()->getFileName();
  }

  QPushButton * fsButton = new QPushButton( fileName.empty() ? "..." : fileName.c_str() );
  fsButton->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
  connect( fsButton, SIGNAL(clicked(bool)), this, SLOT(textureSelectionClicked(bool)) );

  return( fsButton );
}

void ScenePropertiesWidget::updateEdit( QWidget * widget, const dp::sg::core::TextureSharedPtr & value )
{
  std::string fileName;
  if ( value.isPtrTo<dp::sg::core::TextureFile>() )
  {
    fileName = value.staticCast<dp::sg::core::TextureFile>()->getFilename();
  }
  else if ( value.isPtrTo<dp::sg::core::TextureHost>() )
  {
    fileName = value.staticCast<dp::sg::core::TextureHost>()->getFileName();
  }

  DP_ASSERT( dynamic_cast<QPushButton*>(widget) );
  static_cast<QPushButton*>(widget)->setText( fileName.empty() ? "..." : fileName.c_str() );
}

QLayout * ScenePropertiesWidget::createEdit( const dp::math::Quatf & value, dp::util::PropertyId pid, bool enabled )
{
  //DP_ASSERT( !enabled );
  QHBoxLayout * layout = new QHBoxLayout();
  for ( unsigned int i=0 ; i<4 ; i++ )
  {
    QString text = QString( "%1" ).arg( value[i] );
    layout->addWidget( new QLabel( text ) );
  }
  layout->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
  return( layout );
}

void ScenePropertiesWidget::updateEdit( QLayout * layout, const dp::math::Quatf & value )
{
  DP_ASSERT( dynamic_cast<QHBoxLayout*>(layout) );
  for ( unsigned int i=0 ; i<4 ; i++ )
  {
    QString text = QString( "%1" ).arg( value[i] );
    DP_ASSERT( layout->itemAt( i ) && dynamic_cast<QLabel*>(layout->itemAt( i )->widget()) );
    static_cast<QLabel*>(layout->itemAt( i )->widget())->setText( text );
  }
}

QHBoxLayout * ScenePropertiesWidget::createLabledSlider( dp::util::PropertyId pid, float value, float min, float max, unsigned int size, unsigned int index )
{
  int maxLength = static_cast<int>(log10( max ));
  int precision  = std::max( 0, 3 - maxLength );    // three digits precision for max < 10, two for 10 <= max < 100, ...
  int fieldWidth = 1 + precision + maxLength;
  QLabel * label = new QLabel( QString( "%1" ).arg( value, fieldWidth, 'f', precision ) );
  label->setProperty( "FieldWidth", fieldWidth );
  label->setProperty( "Precision", precision );

  QSlider * slider = new QSlider( Qt::Horizontal );
  slider->setMaximum( 1000 );
  DP_ASSERT( slider->minimum() == 0 );
  slider->setValue( slider->maximum() * ( value - min ) / ( max - min ) );
  slider->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
  slider->setProperty( "Min", min );
  slider->setProperty( "Max", max );
  slider->setProperty( "Size", size );
  slider->setProperty( "Index", index );
  slider->setProperty( "Label", QVariant::fromValue( static_cast<void *>( label ) ) );
  connect( slider, SIGNAL(valueChanged(int)), this, SLOT(valueChangedFloat(int)) );

  QHBoxLayout * layout = new QHBoxLayout();
  layout->addWidget( label );
  layout->addWidget( slider );
  layout->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
  return( layout );
}

void ScenePropertiesWidget::updateLabledSlider( QHBoxLayout * layout, float value )
{
  DP_ASSERT( layout && layout->itemAt( 1 ) && dynamic_cast<QSlider*>(layout->itemAt( 1 )->widget()) );
  QSlider * slider = static_cast<QSlider*>(layout->itemAt( 1 )->widget());
  float min = slider->property( "Min" ).toFloat();
  float max = slider->property( "Max" ).toFloat();
  DP_ASSERT( ( min <= value ) && ( value <= max ) );
  slider->setValue( slider->maximum() * ( value - min ) / ( max - min ) );
}

bool checkEnabled( dp::sg::core::ObjectSharedPtr const& o, dp::util::PropertyId pid )
{
  std::string propertyName = o->getPropertyName( pid );                             // filter out ...
  return(   ( ( propertyName != "Hints" ) && ( propertyName != "TraversalMask" ) )  // ... "Hints" and "TraversalMask" in dp::sg::core::Object, as they are not supposed to be editable
        &&  (   !o.isPtrTo<dp::sg::core::Camera>()                                  // ... some properties of a dp::sg::core::Camera, which are supposed to be edited by special means
            ||  ( ( propertyName != "Direction" ) && ( propertyName != "Orientation" ) && ( propertyName != "Position" ) && ( propertyName != "UpVector" ) ) ) );
}

void ScenePropertiesWidget::currentItemChanged( dp::sg::core::ObjectSharedPtr current, dp::sg::core::ObjectSharedPtr previous )
{
  displayItem( current );
}

void ScenePropertiesWidget::displayItem( dp::sg::core::ObjectSharedPtr const & object )
{
  QString title = baseTitle;
  if ( object )
  {
    title += QString( " - " ) + QString( object->getName().c_str() );
    title += QString( " (" ) + QString( dp::sg::core::objectCodeToName( object->getObjectCode() ).c_str() );
    if ( object->getObjectCode() == dp::sg::core::ObjectCode::PRIMITIVE )
    {
      title += QString( " : " ) + QString( dp::sg::core::primitiveTypeToName( object.staticCast<dp::sg::core::Primitive>()->getPrimitiveType() ).c_str() );
    }
    title += QString( ")" );
  }
  setWindowTitle( title );

  if ( m_object )
  {
    m_object->detach( &m_objectObserver );
    m_object.reset();
  }

  QWidget * widget = new QWidget();
  if ( object )
  {
    m_object = object;
    m_object->attach( &m_objectObserver );

    std::map<std::string,std::vector<dp::util::PropertyId> > groupedPids;
    unsigned int propertyCount = m_object->getPropertyCount();
    for ( unsigned int i=0 ; i<propertyCount ; i++ )
    {
      dp::util::PropertyId pid = m_object->getProperty( i );
      if ( !isHiddenProperty( pid ) )      // filter out properties marked with Annotation "_anno_hidden"
      {
        groupedPids[getAnnotation( pid, "_anno_inGroup" )].push_back( pid );
      }
    }
    QVBoxLayout * layout = new QVBoxLayout();
    for ( std::map<std::string,std::vector<dp::util::PropertyId> >::const_iterator git = groupedPids.begin() ; git != groupedPids.end() ; ++git )
    {
      QFormLayout * groupLayout = new QFormLayout();
      for ( size_t i=0 ; i<git->second.size() ; i++ )
      {
        dp::util::PropertyId pid = git->second[i];
        QLabel * label = new QLabel( getDisplayName( pid, m_object->getPropertyName( pid ) ).c_str() );
        std::string description = getAnnotation( pid, "_anno_description" );
        if ( !description.empty() )
        {
          label->setStatusTip( description.c_str() );
          label->setToolTip( description.c_str() );
          label->setWhatsThis( description.c_str() );
        }
        bool enabled = checkEnabled( m_object, pid );
        dp::util::Property::Type type = pid->getType();
        switch( type )
        {
          case dp::util::Property::Type::FLOAT :
            groupLayout->addRow( label, createEdit( m_object->getValue<float>( pid ), pid, enabled ) );
            break;
          case dp::util::Property::Type::FLOAT2 :
            groupLayout->addRow( label, createEdit<2>( m_object->getValue<dp::math::Vecnt<2,float> >( pid ), pid, enabled ) );
            break;
          case dp::util::Property::Type::FLOAT3 :
            groupLayout->addRow( label, createEdit<3>( m_object->getValue<dp::math::Vecnt<3,float> >( pid ), pid, enabled ) );
            break;
          case dp::util::Property::Type::FLOAT4 :
            groupLayout->addRow( label, createEdit<4>( m_object->getValue<dp::math::Vecnt<4,float> >( pid ), pid, enabled ) );
            break;
          case dp::util::Property::Type::INT :
            groupLayout->addRow( label, createEdit( m_object->getValue<int>( pid ), pid, enabled ) );
            break;
          case dp::util::Property::Type::UINT :
            groupLayout->addRow( label, createEdit( m_object->getValue<unsigned int>( pid ), pid, enabled ) );
            break;
          case dp::util::Property::Type::QUATERNION_FLOAT :
            groupLayout->addRow( label, createEdit( m_object->getValue<dp::math::Quatf>( pid ), pid, enabled ) );
            break;
          case dp::util::Property::Type::BOOLEAN :
            groupLayout->addRow( label, createEdit( m_object->getValue<bool>( pid ), pid, enabled ) );
            break;
          case dp::util::Property::Type::STRING :
            groupLayout->addRow( label, createEdit( m_object->getValue<std::string>( pid ), pid, enabled ) );
            break;
          case dp::util::Property::Type::TEXTURE :
            groupLayout->addRow( label, createEdit( m_object->getValue<dp::sg::core::TextureSharedPtr>( pid ), pid, enabled ) );
            break;
          default :
            {
              QLabel * emptyEdit = new QLabel();
              emptyEdit->setProperty( "dp::util::PropertyId", QVariant::fromValue( static_cast<void *>( pid ) ) );
              groupLayout->addRow( label, emptyEdit );
            }
            break;
        }
      }

      QGroupBox * groupBox = new QGroupBox( QString( git->first.c_str() ) );
      groupBox->setLayout( groupLayout );

      layout->addWidget( groupBox );
    }
    widget->setLayout( layout );
  }
  QScrollArea * scrollArea = new QScrollArea();
  scrollArea->setWidget( widget );
  setWidget( scrollArea );
}

void ScenePropertiesWidget::updateItem()
{
  DP_ASSERT( m_object );

  DP_ASSERT( dynamic_cast<QScrollArea*>(widget()) );
  DP_ASSERT( static_cast<QScrollArea*>(widget())->widget() );
  DP_ASSERT( dynamic_cast<QVBoxLayout*>(static_cast<QScrollArea*>(widget())->widget()->layout()) );
  QVBoxLayout * layout = static_cast<QVBoxLayout*>(static_cast<QScrollArea*>(widget())->widget()->layout());
  for ( int i=0 ; i<layout->count() ; i++ )
  {
    DP_ASSERT( layout->itemAt( i ) && dynamic_cast<QGroupBox*>(layout->itemAt( i )->widget()) );
    QGroupBox * groupBox = static_cast<QGroupBox*>(layout->itemAt( i )->widget());
    DP_ASSERT( dynamic_cast<QFormLayout*>(groupBox->layout()) );
    QFormLayout * groupLayout = static_cast<QFormLayout*>(groupBox->layout());
    for ( int j=0 ; j<groupLayout->rowCount() ; j++ )
    {
      QLayoutItem * layoutItem = groupLayout->itemAt( j, QFormLayout::FieldRole );
      DP_ASSERT( layoutItem );
      DP_ASSERT( layoutItem->layout() || layoutItem->widget() );
      DP_ASSERT( layoutItem->layout() ? layoutItem->layout()->property( "dp::util::PropertyId" ).isValid() : layoutItem->widget()->property( "dp::util::PropertyId" ).isValid() );
      dp::util::PropertyId pid = static_cast<dp::util::PropertyId>( layoutItem->layout() ? layoutItem->layout()->property( "dp::util::PropertyId" ).value<void*>()
                                                                                         : layoutItem->widget()->property( "dp::util::PropertyId" ).value<void*>() );
      DP_ASSERT( m_object->hasProperty( pid ) );
      dp::util::Property::Type type = pid->getType();
      switch( type )
      {
        case dp::util::Property::Type::FLOAT :
          updateEdit( layoutItem->layout(), m_object->getValue<float>( pid ), pid );
          break;
        case dp::util::Property::Type::FLOAT2 :
          updateEdit<2>( layoutItem->layout(), m_object->getValue<dp::math::Vecnt<2,float> >( pid ), pid );
          break;
        case dp::util::Property::Type::FLOAT3 :
          updateEdit<3>( layoutItem->layout(), m_object->getValue<dp::math::Vecnt<3,float> >( pid ), pid );
          break;
        case dp::util::Property::Type::FLOAT4 :
          updateEdit<4>( layoutItem->layout(), m_object->getValue<dp::math::Vecnt<4,float> >( pid ), pid );
          break;
        case dp::util::Property::Type::INT :
          updateEdit( layoutItem->widget(), m_object->getValue<int>( pid ), pid );
          break;
        case dp::util::Property::Type::UINT :
          updateEdit( layoutItem->widget(), m_object->getValue<unsigned int>( pid ) );
          break;
        case dp::util::Property::Type::QUATERNION_FLOAT :
          updateEdit( layoutItem->layout(), m_object->getValue<dp::math::Quatf>( pid ) );
          break;
        case dp::util::Property::Type::BOOLEAN :
          updateEdit( layoutItem->widget(), m_object->getValue<bool>( pid ) );
          break;
        case dp::util::Property::Type::STRING :
          updateEdit( layoutItem->widget(), m_object->getValue<std::string>( pid ) );
          break;
        case dp::util::Property::Type::TEXTURE :
          updateEdit( layoutItem->widget(), m_object->getValue<dp::sg::core::TextureSharedPtr>( pid ) );
          break;
        default :
          break;
      }
    }
  }
}

void
ScenePropertiesWidget::clear()
{
  setWindowTitle( baseTitle );
  setWidget( nullptr );
}

