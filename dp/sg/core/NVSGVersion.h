// Copyright NVIDIA Corporation 2002-2005
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
/** @file */

#if !defined( DOXYGEN_IGNORE )    //  no need to document the version header
//---------------------------------------------------------------------------
// Version Number
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// - This is the ONLY place to edit the version number.
// - There must be no unnecessary leading zeros in the numbers.
//   (e.g.: don't use '02' - use '2' instead)
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#define NVSG_VER_MAIN        8    // marketing main version
#define NVSG_VER_AUX         0    // marketing auxiliary version

#define NVSG_VER_MAJOR       10
#define NVSG_VER_MINOR       0

#if !defined(NVSG_VER_BUILD)
#define NVSG_VER_BUILD       0
#endif

//#define NVSG_ALPHA
#define NVSG_BETA

#ifdef LINUX
  #define NVSG_BETA
#elif defined _WIN64
  #define NVSG_BETA
#endif

// MaKe STRing helper macro
#define MKSTR(s)    MKSTR_i(s)
#define MKSTR_i(s)  MKSTR_ii(s)
#define MKSTR_ii(s) #s

// conditionally add a leading zero to the single parts of the version string
#define NVSG_VER_MAIN_STR           MKSTR(NVSG_VER_MAIN)
#define NVSG_VER_AUX_STR            MKSTR(NVSG_VER_AUX)
#define NVSG_VER_MAJOR_STR          MKSTR(NVSG_VER_MAJOR)
#define NVSG_VER_MINOR_STR          MKSTR(NVSG_VER_MINOR)
#define NVSG_VER_BUILD_STR          MKSTR(NVSG_VER_BUILD)

#define SDK_VENDOR "NVIDIA"
#define SDK_NAME   "SceniX"

// internal version string
#define _VER_STR   NVSG_VER_MAIN_STR "."\
                   NVSG_VER_AUX_STR "."\
                   NVSG_VER_MAJOR_STR "."\
                   NVSG_VER_MINOR_STR "."\
                   NVSG_VER_BUILD_STR

// no need to update these
//
#ifdef NVSG_ALPHA
  #if !defined(NDEBUG)
    #define VERSION_STR   _VER_STR " alpha" " (DEBUG)"
  #else  //DEBUG
    #define VERSION_STR   _VER_STR " alpha" 
  #endif //DEBUG
#else
  #ifdef NVSG_BETA
    #if !defined(NDEBUG)
      #define VERSION_STR   _VER_STR " beta (DEBUG)"
    #else  //DEBUG
      #define VERSION_STR   _VER_STR " beta"
    #endif //DEBUG
  #else  //BETA
    #if !defined(NDEBUG)
      #define VERSION_STR   _VER_STR " (DEBUG)"
    #else  //DEBUG
      #define VERSION_STR   _VER_STR
    #endif //DEBUG
  #endif //BETA
#endif

#define NVSG_COPYRIGHT   SDK_VENDOR" "SDK_NAME" Version "VERSION_STR"\n© Copyright 2009 NVIDIA Corporation\n";

#endif  //  DOXYGEN_IGNORE
