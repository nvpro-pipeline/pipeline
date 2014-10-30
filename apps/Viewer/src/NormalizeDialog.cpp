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


#include <QDialogButtonBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QPushButton>
#include "NormalizeDialog.h"
#include "Viewer.h"
#include <dp/sg/algorithm/NormalizeTraverser.h>
#include <dp/sg/algorithm/SearchTraverser.h>

using namespace dp::sg::core;

using std::string;
using std::vector;

NormalizeDialog::NormalizeDialog( const SceneSharedPtr & scene, QWidget * parent )
  : QDialog( parent )
  , m_scene( scene )
{
  static string names[] =
  {
    "Position", "Vertex Weight", "Normal", "Color", "Secondary Color", "Fog Coord", "Unused 1", "Unused 2",
    "TexCoord0", "TexCoord1", "TexCoord2", "TexCoord3", "TexCoord4", "TexCoord5", "TexCoord6", "TexCoord7", 
  };

  setWindowTitle( QApplication::translate( VIEWER_APPLICATION_NAME, "Normalize VertexAttributes" ) );

  for ( int i=0 ; i<16 ; i++ )
  {
    m_vaButtons[i] = new QCheckBox( QApplication::translate( VIEWER_APPLICATION_NAME, names[i].c_str() ) );
  }

  QGridLayout * gridLayout = new QGridLayout;
  for ( int i=0 ; i<4 ; i++ )
  {
    for ( int j=0 ; j<4 ; j++ )
    {
      gridLayout->addWidget( m_vaButtons[4*i+j], i, j );
    }
  }

  QGroupBox * optionsBox = new QGroupBox( QApplication::translate( VIEWER_APPLICATION_NAME, "Options" ) );
  optionsBox->setLayout( gridLayout );

  QDialogButtonBox * dbb = new QDialogButtonBox( QDialogButtonBox::Ok | QDialogButtonBox::Cancel );
  dbb->button( QDialogButtonBox::Ok )->setDefault ( true );

  QVBoxLayout * mainLayout = new QVBoxLayout;
  mainLayout->addWidget( optionsBox );
  mainLayout->addWidget( dbb );

  setLayout( mainLayout );
  adjustSize();
  setMinimumSize( size() );
  setMaximumSize( size() );

  // connect everything up
  connect( dbb, SIGNAL(accepted()), this, SLOT(accept()) );
  connect( dbb, SIGNAL(rejected()), this, SLOT(reject()) );

  // disable m_vaButtons without any corresponding data in the scene
  unsigned short vaMask = 0;
  dp::sg::algorithm::SearchTraverser searchTraverser;
  searchTraverser.setClassName( "class dp::sg::core::VertexAttributeSet" );
  searchTraverser.setBaseClassSearch( true );
  searchTraverser.apply( m_scene );
  const vector<ObjectWeakPtr> &vp = searchTraverser.getResults();
  for ( size_t i=0 ; i<vp.size() ; i++ )
  {
    DP_ASSERT( dynamic_cast<VertexAttributeSetWeakPtr>(vp[i]) );
    VertexAttributeSetSharedPtr const& vas = vp[i]->getSharedPtr<VertexAttributeSet>();
    for ( unsigned int attrib = 0 ; attrib<16 ; ++attrib )
    {
      if ( vas->getNumberOfVertexData( attrib ) )
      {
        vaMask |= ( 1 << attrib );
      }
    }
  }
  for ( unsigned int i=1 ; i<16 ; ++i )
  {
    m_vaButtons[i]->setEnabled( vaMask & ( 1 << i ) );
  }
}

NormalizeDialog::~NormalizeDialog()
{
}

void NormalizeDialog::accept()
{
  GetApp()->setOverrideCursor( Qt::WaitCursor );
  {
    dp::sg::algorithm::NormalizeTraverser normalizeTraverser;
    for ( unsigned int i=0 ; i<16 ; i++ )
    {
      if ( m_vaButtons[i]->isChecked() )
      {
        normalizeTraverser.setVertexAttributeIndex( i );
        normalizeTraverser.apply( m_scene );
      }
    }
  }
  GetApp()->restoreOverrideCursor();

  QDialog::accept();
}
