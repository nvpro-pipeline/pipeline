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


#pragma once

#include <QUndoCommand>
#include <QVariant>
#include <dp/sg/core/CoreTypes.h>
#include <dp/sg/core/VertexAttributeSet.h>
#include <dp/util/Flags.h>

class ViewerCommand : public QUndoCommand
{
  protected:
    enum class UpdateFlag
    {
      ITEMMODELS = 0x01,
      MATERIAL   = 0x02,
      SCENE_TREE = 0x04
    };

    typedef dp::util::Flags<UpdateFlag> UpdateMask;

    friend UpdateMask operator|( UpdateFlag bit0, UpdateFlag bit1 );

  public:
    ViewerCommand( UpdateMask updateFlags = UpdateMask(), bool parameterCommand = false, int id = -1 );
    virtual ~ViewerCommand();

    bool isParameterCommand() const;
    virtual int id() const;
    virtual void undo();
    virtual void redo();

  protected:
    virtual bool doUndo() = 0;
    virtual bool doRedo() = 0;
    void update();

  protected:
    bool m_parameterCommand;
    UpdateMask m_updateFlags;
    int m_id;
};

inline ViewerCommand::UpdateMask operator|( ViewerCommand::UpdateFlag bit0, ViewerCommand::UpdateFlag bit1 )
{
  return ViewerCommand::UpdateMask( bit0 ) | bit1;
}

class CommandReplacePipeline : public ViewerCommand
{
  public:
    CommandReplacePipeline( const dp::sg::core::GeoNodeSharedPtr & geoNode, const dp::sg::core::PipelineDataSharedPtr & newEffect );
    ~CommandReplacePipeline();

  protected:
    virtual bool doUndo();
    virtual bool doRedo();

  private:
    dp::sg::core::GeoNodeSharedPtr      m_geoNode;
    dp::sg::core::PipelineDataSharedPtr m_newPipeline;
    dp::sg::core::PipelineDataSharedPtr m_oldPipeline;
};

class CommandGenerateTangentSpace : public ViewerCommand
{
  public:
    CommandGenerateTangentSpace( const dp::sg::core::PrimitiveSharedPtr & primitive );
    ~CommandGenerateTangentSpace();

  protected:
    virtual bool doUndo();
    virtual bool doRedo();

  private:
    dp::sg::core::PrimitiveSharedPtr  m_primitive;
};

class CommandGenerateTextureCoordinates : public ViewerCommand
{
  public:
    CommandGenerateTextureCoordinates( const dp::sg::core::PrimitiveSharedPtr & primitive, dp::sg::core::TextureCoordType tct );
    ~CommandGenerateTextureCoordinates();

  protected:
    virtual bool doUndo();
    virtual bool doRedo();

  private:
    dp::sg::core::PrimitiveSharedPtr  m_primitive;
    dp::sg::core::TextureCoordType    m_tct;
};
