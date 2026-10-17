import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import solc from 'solc';
import ganache from 'ganache';
import { BrowserProvider, ContractFactory, id, ZeroHash } from 'ethers';

const source = await fs.readFile('contracts/AnalysisRegistry.sol', 'utf8');
const compiled = JSON.parse(solc.compile(JSON.stringify({
  language:'Solidity', sources:{'AnalysisRegistry.sol':{content:source}},
  settings:{evmVersion:'shanghai',outputSelection:{'*':{'*':['abi','evm.bytecode.object']}}}
})));
const errors=(compiled.errors || []).filter(error=>error.severity==='error');
assert.deepEqual(errors,[]);
const artifact=compiled.contracts['AnalysisRegistry.sol'].AnalysisRegistry;
// In-process chain with throwaway accounts. No public RPC or paid transaction.
const chain=ganache.provider({logging:{quiet:true},chain:{chainId:1337,hardfork:'shanghai'},wallet:{totalAccounts:2}});
try {
  const provider=new BrowserProvider(chain);
  const owner=await provider.getSigner(0), stranger=await provider.getSigner(1);
  const contract=await new ContractFactory(artifact.abi,artifact.evm.bytecode.object,owner).deploy();
  await contract.waitForDeployment();
  const fileHash=id('file'), receiptHash=id('signed receipt');
  assert.equal(await contract.publisher(),await owner.getAddress());
  const rejectsTransaction = (operation) => assert.rejects(async () => { await (await operation()).wait(); });
  await rejectsTransaction(()=>contract.connect(stranger).record(fileHash,receiptHash));
  await rejectsTransaction(()=>contract.record(ZeroHash,receiptHash));
  await rejectsTransaction(()=>contract.record(fileHash,ZeroHash));
  await (await contract.record(fileHash,receiptHash)).wait();
  const timestamp=await contract.recordedAt(fileHash,receiptHash);
  assert(timestamp>0n);
  await rejectsTransaction(()=>contract.record(fileHash,receiptHash));
  assert.equal(await contract.recordedAt(id('other file'),receiptHash),0n);
  assert.equal(await contract.recordedAt(fileHash,id('altered receipt')),0n);
  await (await contract.record(fileHash,id('second model receipt'))).wait();
  assert.equal(await contract.recordedAt(fileHash,receiptHash),timestamp);
  const report={status:'passed',chain:'in-process Ganache',chainId:1337,compiler:solc.version(),
    checks:['compilation','immutable publisher','unauthorized writer rejected','zero commitments rejected',
    'immutable receipt registration','duplicate rejected','changed file rejected','changed receipt rejected','multiple analyses preserved'],
    publicChainDeployed:false};
  await fs.mkdir('reports',{recursive:true});
  await fs.writeFile('reports/contract-validation.json',JSON.stringify(report,null,2));
  console.log(JSON.stringify(report));
} finally { await chain.disconnect(); }
