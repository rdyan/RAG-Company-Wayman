import { useEffect, useState } from 'react'
import { Typography, Table, Button, Space, Spin, Alert, Upload, message, Popconfirm, Modal } from 'antd'
import { UploadOutlined, DeleteOutlined, ReloadOutlined, EyeOutlined } from '@ant-design/icons'
import axios from 'axios'

const { Title } = Typography

const apiClient = axios.create({ baseURL: 'http://localhost:8000' })

function ArchivesPage() {
  const [files, setFiles] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedRowKeys, setSelectedRowKeys] = useState([])
  const [uploading, setUploading] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [viewing, setViewing] = useState(false)
  const [viewContent, setViewContent] = useState('')
  const [viewError, setViewError] = useState('')
  const [viewFile, setViewFile] = useState('')

  // 获取档案列表
  const fetchFiles = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await apiClient.get('/api/archives/list')
      setFiles(res.data.files || [])
    } catch (e) {
      setError('获取档案列表失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchFiles()
  }, [])

  // 上传文件
  const handleUpload = async ({ file, onSuccess, onError }) => {
    const formData = new FormData()
    formData.append('files', file)
    setUploading(true)
    try {
      await apiClient.post('/api/archives/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      onSuccess()
      message.success(`${file.name} 上传成功`)
      fetchFiles()
    } catch (e) {
      onError()
      message.error(`${file.name} 上传失败`)
    } finally {
      setUploading(false)
    }
  }

  // 批量上传
  const customRequest = (options) => {
    handleUpload(options)
  }

  // 单个删除
  const handleDelete = async (fileName) => {
    setDeleting(true)
    try {
      await apiClient.delete('/api/archives/delete', { params: { file_name: fileName } })
      message.success(`已删除 ${fileName}`)
      fetchFiles()
      setSelectedRowKeys((prev) => prev.filter((k) => k !== fileName))
    } catch (e) {
      message.error(`删除失败: ${fileName}`)
    } finally {
      setDeleting(false)
    }
  }

  // 批量删除
  const handleBatchDelete = async () => {
    setDeleting(true)
    try {
      await Promise.all(selectedRowKeys.map(fn => apiClient.delete('/api/archives/delete', { params: { file_name: fn } })))
      message.success('批量删除成功')
      fetchFiles()
      setSelectedRowKeys([])
    } catch (e) {
      message.error('批量删除失败')
    } finally {
      setDeleting(false)
    }
  }

  // 查看内容
  const handleView = async (fileName) => {
    setViewing(true)
    setViewContent('')
    setViewError('')
    setViewFile(fileName)
    try {
      const res = await apiClient.get('/api/archives/view', { params: { file_name: fileName } })
      setViewContent(res.data.content)
    } catch (e) {
      setViewError('获取内容失败')
    }
  }

  // 获取文件扩展名
  const getFileExt = (fileName) => {
    const idx = fileName.lastIndexOf('.')
    return idx !== -1 ? fileName.slice(idx + 1).toLowerCase() : ''
  }

  // 构造静态文件URL
  const getStaticUrl = (fileName) => `http://localhost:8000/static/archives/${encodeURIComponent(fileName)}`

  const columns = [
    { title: '文件名', dataIndex: 'name', key: 'name', render: (text) => text },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space>
          <Button icon={<EyeOutlined />} onClick={() => handleView(record.name)}>查看</Button>
          <Popconfirm title={`确定删除 ${record.name} 吗？`} onConfirm={() => handleDelete(record.name)} okText="删除" cancelText="取消">
            <Button type="link" icon={<DeleteOutlined />} danger loading={deleting}>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  const rowSelection = {
    selectedRowKeys,
    onChange: setSelectedRowKeys,
  }

  return (
    <div style={{ padding: 32 }}>
      <Title level={3}>档案管理</Title>
      <Space style={{ marginBottom: 16 }}>
        <Upload
          customRequest={customRequest}
          multiple
          showUploadList={false}
          disabled={uploading}
        >
          <Button icon={<UploadOutlined />} loading={uploading}>批量上传</Button>
        </Upload>
        <Button icon={<ReloadOutlined />} onClick={fetchFiles}>刷新列表</Button>
        <Popconfirm title={`确定删除选中的${selectedRowKeys.length}个档案吗？`} onConfirm={handleBatchDelete} okText="删除" cancelText="取消" disabled={selectedRowKeys.length === 0}>
          <Button icon={<DeleteOutlined />} danger disabled={selectedRowKeys.length === 0} loading={deleting}>批量删除</Button>
        </Popconfirm>
      </Space>
      {error && <Alert message={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      <Spin spinning={loading || uploading || deleting} tip="处理中...">
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
        title={viewFile ? `查看内容：${viewFile}` : ''}
        onCancel={() => { setViewFile(''); setViewContent(''); setViewError('') }}
        footer={null}
        width={800}
        styles={{ body: { maxHeight: 500, overflowY: 'auto', background: '#fafafa' } }}
      >
        {viewing && !viewContent && !viewError && <Spin tip="加载中..." />}
        {viewError && <Alert message={viewError} type="error" showIcon />}
        {/* 原格式预览逻辑 */}
        {viewFile && !viewError && (
          (() => {
            const ext = getFileExt(viewFile)
            const url = getStaticUrl(viewFile)
            if (["pdf"].includes(ext)) {
              return <iframe src={url} title="pdf预览" width="100%" height="500px" style={{ border: 0 }} />
            } else if (["jpg", "jpeg", "png", "gif", "webp", "bmp", "svg"].includes(ext)) {
              return <img src={url} alt={viewFile} style={{ maxWidth: '100%', maxHeight: 480, display: 'block', margin: '0 auto' }} />
            } else if (["txt", "json"].includes(ext)) {
              return viewContent ? (
                <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: 14, margin: 0 }}>{viewContent}</pre>
              ) : null
            } else {
              return <a href={url} download style={{ fontSize: 16 }}>下载原文件</a>
            }
          })()
        )}
      </Modal>
    </div>
  )
}

export default ArchivesPage 