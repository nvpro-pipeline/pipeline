// Copyright NVIDIA Corporation 2013
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

#include <QDockWidget>
#include <QTreeWidget>
#include <dp/sg/core/Object.h>
#include <dp/sg/core/Path.h>

class SceneTreeBrowser : public QDockWidget
{
  Q_OBJECT

  public:
    SceneTreeBrowser( QWidget * parent );
    virtual ~SceneTreeBrowser();

  public:
    QTreeWidget * getTree() const;
    void setScene( dp::sg::core::SceneSharedPtr const & scene );

  signals:
    void currentItemChanged( dp::sg::core::ObjectSharedPtr current, dp::sg::core::ObjectSharedPtr previous );

  protected slots:
    void currentItemChanged( QTreeWidgetItem * current, QTreeWidgetItem * previous );
    void itemExpanded( QTreeWidgetItem * item );
    void itemPressed( QTreeWidgetItem * item, int column );
    void selectObject( dp::sg::core::PathSharedPtr const& path );
    void triggeredAddHeadlightMenu( QAction * action );
    void triggeredAddSamplerMenu( QAction * action );
    void triggeredDeleteObject();
    void triggeredGenerateTangentSpace();
    void triggeredGenerateTextureCoordinatesMenu( QAction * action );
    void triggeredReplaceByClone();
    void triggeredSaveEffectData();
    void triggeredShowShaderPipeline();
    void updateTree();

  protected:
    void contextMenuEvent( QContextMenuEvent * event );

  private:
    class ObjectObserver : public dp::util::Observer
    {
      public:
        ObjectObserver( SceneTreeBrowser * stb );

        virtual void onNotify( const dp::util::Event &event, dp::util::Payload *payload );
        virtual void onDestroyed( const dp::util::Subject& subject, dp::util::Payload* payload );

      private:
        SceneTreeBrowser * m_stb;
    };

  private:
    QTreeWidget     * m_tree;
    ObjectObserver    m_objectObserver;
};

