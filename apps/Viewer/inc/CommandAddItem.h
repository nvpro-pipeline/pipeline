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


#include "SceneTreeItem.h"
#include "ViewerCommand.h"

class CommandAddItem : public ViewerCommand
{
  public:
    CommandAddItem( SceneTreeItem * parent, SceneTreeItem * child, bool add = true );
    virtual ~CommandAddItem();

  protected:
    virtual bool doUndo();
    virtual bool doRedo();

  private:
    void adjustItems( bool add );

  private:
    bool            m_add;
    SceneTreeItem * m_parent;
    SceneTreeItem * m_child;
    bool            m_childTaken;
};

class CommandAddObject : public ViewerCommand
{
  public:
    CommandAddObject( dp::sg::core::ObjectSharedPtr const& parent, dp::sg::core::ObjectSharedPtr const& child, bool add = true );
    virtual ~CommandAddObject();

  protected:
    virtual bool doUndo();
    virtual bool doRedo();

  private:
    bool                          m_add;
    dp::sg::core::ObjectSharedPtr m_parent;
    dp::sg::core::ObjectSharedPtr m_child;
};
