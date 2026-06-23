/*********************                                                        */
/*! \file File.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __File_h__
#define __File_h__

class ConstSimpleData;
class HeapData;

#include "IFile.h"
#include "MString.h"

#ifdef _WIN32
#include <io.h>
#define S_ISDIR( mode ) ( ( (mode)&S_IFMT ) == S_IFDIR )

typedef int mode_t;

static const mode_t S_ISUID = 0x08000000;
static const mode_t S_ISGID = 0x04000000;
static const mode_t S_ISVTX = 0x02000000;
static const mode_t S_IRUSR = mode_t( _S_IREAD );
static const mode_t S_IWUSR = mode_t( _S_IWRITE );
static const mode_t S_IXUSR = 0x00400000;
#ifndef STRICT_UGO_PERMISSIONS
// without STRICT_UGO_PERMISSIONS, group/other map to user
static const mode_t S_IRGRP = mode_t( _S_IREAD );
static const mode_t S_IWGRP = mode_t( _S_IWRITE );
static const mode_t S_IXGRP = 0x00080000;
static const mode_t S_IROTH = mode_t( _S_IREAD );
static const mode_t S_IWOTH = mode_t( _S_IWRITE );
static const mode_t S_IXOTH = 0x00010000;
#else
static const mode_t S_IRGRP = 0x00200000;
static const mode_t S_IWGRP = 0x00100000;
static const mode_t S_IXGRP = 0x00080000;
static const mode_t S_IROTH = 0x00040000;
static const mode_t S_IWOTH = 0x00020000;
static const mode_t S_IXOTH = 0x00010000;
#endif
#endif

class File : public IFile
{
public:
    File( const String &path );
    virtual ~File();
    void close();
    static bool directory( const String &path );
    static unsigned getSize( const String &path );
    void open( Mode openMode );
    void write( const String &line );
    void write( const ConstSimpleData &data );
    void read( HeapData &buffer, unsigned maxReadSize );
    String readLine( char lineSeparatingChar = '\n' );

private:
    enum {
        NO_DESCRIPTOR = -1,
    };

    String _path;
    int _descriptor;
    String _readLineBuffer;

    void closeIfNeeded();
};

#endif // __File_h__
