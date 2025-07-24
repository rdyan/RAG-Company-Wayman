import { useEffect, useState } from 'react'
import { Typography, Table, Button, Space, Spin, Alert, InputNumber, Tag, Modal, Collapse, Card } from 'antd'
import axios from 'axios'

const { Title } = Typography

const apiClient = axios.create({ baseURL: 'http://localhost:8000' })

function AuditPage() {
  const [files, setFiles] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedRowKeys, setSelectedRowKeys] = useState([])
  const [topK, setTopK] = useState(20)
  const [rerankTopN, setRerankTopN] = useState(5)
  const [auditing, setAuditing] = useState(false)
  const [statusMap, setStatusMap] = useState({}) // {fileName: 'pending'|'processing'|'done'|'error'}
  const [resultMap, setResultMap] = useState({}) // {fileName: result}
  const [viewFile, setViewFile] = useState(null)

  // 拉取所有历史审核结果并初始化状态
  const fetchAllAuditResults = async (fileList) => {
    try {
      const res = await apiClient.get('/api/audit/all_results')
      const allResults = res.data.all_results || []
      // 合并所有历史结果，取每个文件最新一次
      const latestResultMap = {}
      const latestStatusMap = {}
      for (const batch of allResults) {
        for (const item of batch.results) {
          const fn = item.file_name
          if (!(fn in latestResultMap)) {
            latestResultMap[fn] = item.result
            if (item.result && Array.isArray(item.result) && item.result.length > 0 && !item.result[0].error) {
              latestStatusMap[fn] = 'done'
            } else {
              latestStatusMap[fn] = 'error'
            }
          }
        }
      }
      // 初始化所有文件状态
      const status = {}
      const result = {}
      fileList.forEach(f => {
        if (f in latestStatusMap) {
          status[f] = latestStatusMap[f]
          result[f] = latestResultMap[f]
        } else {
          status[f] = 'pending'
        }
      })
      setStatusMap(status)
      setResultMap(result)
    } catch (e) {
      // 如果拉取失败，全部初始化为pending
      const status = {}
      fileList.forEach(f => { status[f] = 'pending' })
      setStatusMap(status)
      setResultMap({})
    }
  }

  const fetchFiles = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await apiClient.get('/api/archives/list')
      setFiles(res.data.files || [])
      await fetchAllAuditResults(res.data.files || [])
    } catch (e) {
      setError('获取档案列表失败')
    } finally {
      setLoading(false)
    }
  }

  // 状态和结果持久化
  useEffect(() => {
    // 每次状态或结果变化时保存
    localStorage.setItem('auditStatusMap', JSON.stringify(statusMap))
    localStorage.setItem('auditResultMap', JSON.stringify(resultMap))
  }, [statusMap, resultMap])

  useEffect(() => {
    fetchFiles()
  }, [])

  // 批量审核
  const handleAudit = async () => {
    // 标记选中为processing
    const newStatus = { ...statusMap }
    selectedRowKeys.forEach(f => { newStatus[f] = 'processing' })
    setStatusMap(newStatus)
    setAuditing(true)
    try {
      const res = await apiClient.post('/api/audit', {
        file_names: selectedRowKeys,
        top_k: topK,
        rerank_top_n: rerankTopN,
      })
      // 更新状态和结果
      const newResultMap = { ...resultMap }
      const newStatusMap = { ...statusMap }
      res.data.results.forEach(item => {
        if (item.result && Array.isArray(item.result) && item.result.length > 0 && !item.result[0].error) {
          newStatusMap[item.file_name] = 'done'
        } else {
          newStatusMap[item.file_name] = 'error'
        }
        newResultMap[item.file_name] = item.result
      })
      setResultMap(newResultMap)
      setStatusMap(newStatusMap)
    } catch (e) {
      // 全部失败
      const newStatusMap = { ...statusMap }
      selectedRowKeys.forEach(f => { newStatusMap[f] = 'error' })
      setStatusMap(newStatusMap)
    } finally {
      setAuditing(false)
    }
  }

  const columns = [
    { title: '文件名', dataIndex: 'name', key: 'name', render: (text) => text },
    { title: '状态', dataIndex: 'name', key: 'status', render: (name) => {
      const status = statusMap[name] || 'pending'
      if (status === 'done') return <Tag color="green">已审核</Tag>
      if (status === 'processing') return <Tag color="blue">审核中</Tag>
      if (status === 'error') return <Tag color="red">失败</Tag>
      return <Tag>未审核</Tag>
    }},
    { title: '操作', dataIndex: 'name', key: 'action', render: (name) => {
      const status = statusMap[name] || 'pending'
      if (status === 'done' || status === 'error') {
        return <Button size="small" onClick={() => setViewFile(name)}>查看结果</Button>
      }
      return null
    }},
  ]

  const rowSelection = {
    selectedRowKeys,
    onChange: setSelectedRowKeys,
    getCheckboxProps: (record) => ({ disabled: statusMap[record.name] === 'processing' })
  }

  return (
    <div style={{ padding: 32 }}>
      <Title level={3}>批量审核</Title>
      <Space style={{ marginBottom: 16 }}>
        <span>top_k:</span>
        <InputNumber min={1} max={100} value={topK} onChange={setTopK} />
        <span>rerank_top_n:</span>
        <InputNumber min={1} max={topK} value={rerankTopN} onChange={setRerankTopN} />
        <Button type="primary" disabled={selectedRowKeys.length === 0 || auditing} loading={auditing} onClick={handleAudit}>
          批量审核
        </Button>
      </Space>
      {error && <Alert message={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      <Spin spinning={loading} tip="加载中...">
        <Table
          rowKey="name"
          dataSource={files.map(name => ({ name }))}
          columns={columns}
          rowSelection={rowSelection}
          pagination={false}
        />
      </Spin>
      <Modal
        open={!!viewFile}
        title={viewFile ? `审核结果：${viewFile}` : ''}
        onCancel={() => setViewFile(null)}
        footer={null}
        width={800}
        bodyStyle={{ maxHeight: 500, overflowY: 'auto', background: '#fafafa' }}
      >
        {viewFile && resultMap[viewFile] && (
          <Collapse defaultActiveKey={resultMap[viewFile].map((_, i) => i.toString())}>
            {resultMap[viewFile].map((seg, idx) => (
              <Collapse.Panel header={`分段${idx+1}`} key={idx}>
                {seg.error ? (
                  <Alert message={seg.error} type="error" showIcon />
                ) : (
                  <Card size="small" style={{ marginBottom: 8 }}>
                    <div style={{ fontWeight: 'bold', marginBottom: 8 }}>审核结果：</div>
                    <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: 14, margin: 0 }}>{JSON.stringify(seg, null, 2)}</pre>
                  </Card>
                )}
              </Collapse.Panel>
            ))}
          </Collapse>
        )}
      </Modal>
    </div>
  )
}

export default AuditPage
 