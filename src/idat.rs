use byteorder::{LittleEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct DirEntry { pub code: u16, pub offset: u64 }

#[derive(Debug)]
pub struct IdatData {
    pub version: u32,
    pub nfields: u32,
    pub nsnps: u32,
    pub illumina_ids: Vec<u32>,
    pub means: Vec<u16>,
}

const FID_NSNP: u16 = 1000;
const FID_ILMNID: u16 = 102;
const FID_MEAN: u16 = 104;

fn read_u16<R: Read>(r: &mut R) -> Result<u16, String> { r.read_u16::<LittleEndian>().map_err(|e| e.to_string()) }
fn read_u32<R: Read>(r: &mut R) -> Result<u32, String> { r.read_u32::<LittleEndian>().map_err(|e| e.to_string()) }
fn read_u64<R: Read>(r: &mut R) -> Result<u64, String> { r.read_u64::<LittleEndian>().map_err(|e| e.to_string()) }

fn read_block_u32<R: Read + Seek>(r: &mut R, offset: u64, n: usize) -> Result<Vec<u32>, String> {
    r.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n { out.push(read_u32(r)?); }
    Ok(out)
}
fn read_block_u16<R: Read + Seek>(r: &mut R, offset: u64, n: usize) -> Result<Vec<u16>, String> {
    r.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n { out.push(r.read_u16::<LittleEndian>().map_err(|e| e.to_string())?); }
    Ok(out)
}

pub fn read_idat_from_reader<R: Read + Seek>(mut reader: R) -> Result<IdatData, String> {
    // Magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"IDAT" { return Err("bad magic".into()); }
    // Illumina IDAT v3 directory-style header:
    // u64 version (expect 3), u32 nfields, then nfields x (u16 code, u64 offset)
    let version = read_u64(&mut reader)? as u32; // == 3
    if version != 3 { return Err(format!("unsupported version {}", version)); }
    let nfields = read_u32(&mut reader)?;
    if nfields == 0 || nfields > 50_000 { return Err(format!("unreasonable nfields {}", nfields)); }

    let mut entries = Vec::with_capacity(nfields as usize);
    for _ in 0..nfields {
        let code = read_u16(&mut reader)?;
        let offset = read_u64(&mut reader)?;
        entries.push(DirEntry { code, offset });
    }
    let find = |c: u16| entries.iter().find(|e| e.code == c).ok_or_else(|| format!("missing field {}", c));
    let e_n = find(FID_NSNP)?; let e_ids = find(FID_ILMNID)?; let e_mean = find(FID_MEAN)?;

    reader.seek(SeekFrom::Start(e_n.offset)).map_err(|e| e.to_string())?;
    let nsnps = read_u32(&mut reader)?;
    let illumina_ids = read_block_u32(&mut reader, e_ids.offset, nsnps as usize)?;
    let means = read_block_u16(&mut reader, e_mean.offset, nsnps as usize)?;
    Ok(IdatData { version, nfields, nsnps, illumina_ids, means })
}

pub fn read_idat_any(path: &Path) -> Result<IdatData, String> {
    match path.extension().and_then(|s| s.to_str()) {
        Some("gz") => {
            let f = File::open(path).map_err(|e| e.to_string())?;
            let mut gz = GzDecoder::new(f);
            let mut buf = Vec::new();
            gz.read_to_end(&mut buf).map_err(|e| e.to_string())?;
            read_idat_from_reader(Cursor::new(buf))
        }
        _ => read_idat_from_reader(File::open(path).map_err(|e| e.to_string())?),
    }
}
