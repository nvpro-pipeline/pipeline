// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <dp/fx/mdl/inc/MDLTokenizer.h>
#include <tinyxml.h>

class XMLBuilder : public dp::fx::mdl::MDLTokenizer
{
  public:
    XMLBuilder();
    TiXmlElement * getXMLTree();

  protected:
    bool annotationBegin( std::string const& name );
    void annotationEnd();
    bool argumentBegin( unsigned int idx, std::string const& type, std::string const& name );
    void argumentEnd();
    bool arrayBegin( std::string const& type, size_t size );
    void arrayEnd();
    bool callBegin( std::string const& type, std::string const& name );
    void callEnd();
    void defaultRef( std::string const& type );
    bool enumTypeBegin( std::string const& name, size_t size );
    void enumTypeEnd();
    void enumTypeValue( std::string const& name, int value );
    bool fieldBegin( std::string const& name );
    void fieldEnd();
    bool fileBegin( std::string const& name );
    void fileEnd();
    bool materialBegin( std::string const& name, dp::math::Vec4ui const& hash );
    void materialEnd();
    bool matrixBegin( std::string const& type );
    void matrixEnd();
    bool parameterBegin( unsigned int index, std::string const& name );
    void parameterEnd();
    void referenceParameter( unsigned int idx );
    void referenceTemporary( unsigned int idx );
    bool structureBegin( std::string const& type );
    void structureEnd();
    bool structureTypeBegin( std::string const& name );
    void structureTypeElement( std::string const& type, std::string const& name );
    void structureTypeEnd();
    bool temporaryBegin( unsigned int idx );
    void temporaryEnd();
    void valueBool( bool value );
    void valueBsdfMeasurement( std::string const& value );
    void valueColor( dp::math::Vec3f const& value );
    void valueEnum( std::string const& type, int value, std::string const& name );
    void valueFloat( float value );
    void valueInt( int value );
    void valueLightProfile( std::string const& value );
    void valueString( std::string const& value );
    void valueTexture( std::string const& name, GammaMode gamma );
    bool vectorBegin( std::string const& type );
    void vectorEnd();

  private:
    void popElement();

  private:
    std::string                             m_currentType;
    TiXmlElement *                          m_enumElement;
    std::set<std::string>                   m_enums;
    std::string                             m_fileName;
    TiXmlElement *                          m_libraryElement;
    std::vector<std::stack<TiXmlElement *>> m_materialElements;
    std::set<std::string>                   m_structures;
    std::stack<TiXmlElement *>              m_structureStack;
    std::map<unsigned int,unsigned int>     m_temporaryToParameterMap;
};
